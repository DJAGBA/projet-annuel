"""Injection d'echantillons synthetiques dans le train, et rien que dans le train.

Ce module est la frontiere entre le GAN d'augmentation (perimetre ML) et le jeu
de donnees d'evaluation (perimetre data engineering). Il ne genere rien : il
recoit des fraudes synthetiques deja produites, les valide, les injecte dans le
train et **prouve** que le test n'en contient aucune.

La regle qu'il fait respecter tient en une phrase : *le test doit rester 100 %
reel*. La violer ne provoque aucune erreur visible, seulement des metriques
flatteuses -- un modele evalue en partie sur des donnees issues de son propre
generateur se note lui-meme.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data import config
from src.data.splits import class_balance
from src.data.utils import hash_array

__all__ = [
    "AugmentedTrainingSet",
    "inject_synthetic",
    "row_signatures",
    "assert_no_synthetic_in_test",
    "assert_rows_are_disjoint",
    "assert_test_unchanged",
    "save_synthetic",
    "load_synthetic",
]


# ---------------------------------------------------------------------------
# Signatures de lignes
# ---------------------------------------------------------------------------


def row_signatures(X: np.ndarray) -> set[bytes]:
    """Signature binaire exacte de chaque ligne d'une matrice.

    La comparaison est exacte (octet a octet) et non approchee : deux lignes
    issues d'un tirage continu ne coincident jamais par hasard. Une egalite
    signale donc une **copie**, c'est-a-dire une fuite reelle, jamais une
    coincidence numerique.

    Args:
        X: Matrice `(n_samples, n_features)`.

    Returns:
        L'ensemble des signatures de lignes.
    """
    X = np.ascontiguousarray(X)
    return {row.tobytes() for row in X}


def assert_rows_are_disjoint(
    left: np.ndarray,
    right: np.ndarray,
    *,
    left_name: str = "left",
    right_name: str = "right",
) -> None:
    """Verifie qu'aucune ligne n'est commune a deux matrices.

    Args:
        left: Premiere matrice.
        right: Seconde matrice.
        left_name: Nom de la premiere matrice, pour le message d'erreur.
        right_name: Nom de la seconde matrice.

    Raises:
        AssertionError: Si au moins une ligne est partagee.
    """
    if left.size == 0 or right.size == 0:
        return

    shared = row_signatures(left) & row_signatures(right)
    assert not shared, (
        f"Fuite : {len(shared)} ligne(s) presente(s) a la fois dans "
        f"{left_name} et {right_name}."
    )


def assert_no_synthetic_in_test(X_test: np.ndarray, X_synthetic: np.ndarray) -> None:
    """Garantit que le jeu de test ne contient aucun echantillon synthetique.

    C'est le garde-fou central de la phase d'augmentation. A appeler apres
    chaque injection, et avant toute evaluation.

    Args:
        X_test: Jeu de test, cense etre integralement reel.
        X_synthetic: Echantillons generes.

    Raises:
        AssertionError: Si une ligne synthetique se retrouve dans le test.
    """
    assert_rows_are_disjoint(
        X_test, X_synthetic, left_name="le test", right_name="les donnees synthetiques"
    )


def assert_test_unchanged(expected: np.ndarray, actual: np.ndarray) -> None:
    """Verifie qu'un jeu de test n'a pas bouge entre deux moments du pipeline.

    Args:
        expected: Test de reference (avant augmentation).
        actual: Test observe (apres augmentation).

    Raises:
        AssertionError: Si la forme ou le contenu different.
    """
    assert expected.shape == actual.shape, (
        f"Le test a change de forme : {expected.shape} -> {actual.shape}."
    )
    assert hash_array(expected) == hash_array(actual), (
        "Le contenu du test a ete modifie ; il doit rester intact et 100 % reel."
    )


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AugmentedTrainingSet:
    """Train augmente, avec tracabilite de l'origine de chaque ligne.

    Attributes:
        X: Features, reelles et synthetiques melangees.
        y: Etiquettes alignees sur `X`.
        is_synthetic: Masque booleen, `True` pour les lignes generees. C'est ce
            masque qui rend l'augmentation auditable apres coup : sans lui,
            l'origine des lignes est perdue des le melange.
        seed: Graine du melange.
    """

    X: np.ndarray
    y: np.ndarray
    is_synthetic: np.ndarray
    seed: int

    @property
    def n_real(self) -> int:
        """Nombre de lignes reelles."""
        return int((~self.is_synthetic).sum())

    @property
    def n_synthetic(self) -> int:
        """Nombre de lignes synthetiques."""
        return int(self.is_synthetic.sum())

    @property
    def synthetic_ratio(self) -> float:
        """Part des lignes synthetiques dans le train augmente."""
        return float(self.n_synthetic / self.X.shape[0]) if self.X.size else 0.0

    def real_subset(self) -> tuple[np.ndarray, np.ndarray]:
        """Reconstitue le train reel d'origine (hors ordre)."""
        return self.X[~self.is_synthetic], self.y[~self.is_synthetic]

    def synthetic_subset(self) -> tuple[np.ndarray, np.ndarray]:
        """Extrait les seules lignes generees."""
        return self.X[self.is_synthetic], self.y[self.is_synthetic]

    def summary(self) -> dict[str, object]:
        """Resume serialisable en JSON, a joindre aux metadonnees d'un run."""
        return {
            "n_total": int(self.X.shape[0]),
            "n_real": self.n_real,
            "n_synthetic": self.n_synthetic,
            "synthetic_ratio": self.synthetic_ratio,
            "balance": class_balance(self.y),
            "seed": self.seed,
        }


def inject_synthetic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_synthetic: np.ndarray,
    *,
    label: int = config.FRAUD_LABEL,
    seed: int = config.SEED,
    shuffle: bool = True,
) -> AugmentedTrainingSet:
    """Ajoute des echantillons synthetiques au train, jamais au test.

    Cette fonction ne recoit deliberement **aucun** jeu de test : elle est dans
    l'incapacite structurelle de le contaminer. Le garde-fou
    `assert_no_synthetic_in_test` reste disponible pour verifier l'invariant en
    aval, une fois les deux jeux reunis pour l'evaluation.

    Args:
        X_train: Features d'entrainement reelles.
        y_train: Etiquettes d'entrainement reelles.
        X_synthetic: Fraudes generees. Un tableau vide est accepte et laisse le
            train inchange a l'ordre pres : c'est le run de controle
            (0 % d'augmentation) auquel les runs augmentes seront compares.
        label: Etiquette affectee aux lignes generees.
        seed: Graine du melange.
        shuffle: Melange le resultat. Sans melange, toutes les lignes
            synthetiques se retrouvent en fin de tableau et les derniers batchs
            d'une epoque seraient exclusivement synthetiques. Le melange
            s'applique aussi au run de controle, pour que l'ordre des lignes ne
            soit pas une difference de traitement entre les deux protocoles.

    Returns:
        L'`AugmentedTrainingSet` correspondant.

    Raises:
        ValueError: Si les dimensions sont incoherentes, ou si les echantillons
            synthetiques contiennent des valeurs non finies (cas classique d'un
            generateur qui a diverge).
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_synthetic = np.asarray(X_synthetic)

    if X_train.ndim != 2:
        raise ValueError(f"X_train doit etre 2-D, recu une forme {X_train.shape}.")
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"X_train ({X_train.shape[0]}) et y_train ({y_train.shape[0]}) "
            "ne sont pas alignes."
        )

    if X_synthetic.size == 0:
        combined_X, combined_y = X_train, y_train
        is_synthetic = np.zeros(X_train.shape[0], dtype=bool)
    else:
        if X_synthetic.ndim != 2:
            raise ValueError(
                f"X_synthetic doit etre 2-D, recu une forme {X_synthetic.shape}."
            )
        if X_synthetic.shape[1] != X_train.shape[1]:
            raise ValueError(
                f"X_synthetic a {X_synthetic.shape[1]} features, "
                f"le train en attend {X_train.shape[1]}."
            )
        # Un WGAN-GP qui diverge emet des NaN/inf : sans ce controle, ils
        # polluent silencieusement le train et font echouer l'entrainement
        # bien plus loin, avec une cause introuvable.
        n_invalid = int((~np.isfinite(X_synthetic)).any(axis=1).sum())
        if n_invalid:
            raise ValueError(
                f"{n_invalid} echantillon(s) synthetique(s) contiennent des valeurs "
                "non finies (NaN/inf). Le generateur a probablement diverge."
            )

        combined_X = np.concatenate([X_train, X_synthetic], axis=0)
        combined_y = np.concatenate(
            [y_train, np.full(X_synthetic.shape[0], label, dtype=y_train.dtype)]
        )
        is_synthetic = np.concatenate(
            [
                np.zeros(X_train.shape[0], dtype=bool),
                np.ones(X_synthetic.shape[0], dtype=bool),
            ]
        )

    if shuffle and combined_X.shape[0]:
        order = np.random.default_rng(seed).permutation(combined_X.shape[0])
        combined_X = combined_X[order]
        combined_y = combined_y[order]
        is_synthetic = is_synthetic[order]

    return AugmentedTrainingSet(
        X=combined_X, y=combined_y, is_synthetic=is_synthetic, seed=seed
    )


# ---------------------------------------------------------------------------
# Echange avec le perimetre ML : lecture / ecriture des echantillons generes
# ---------------------------------------------------------------------------


def save_synthetic(
    X_synthetic: np.ndarray,
    name: str,
    directory: Path = config.DATA_SYNTHETIC,
) -> Path:
    """Persiste un lot d'echantillons generes.

    Args:
        X_synthetic: Matrice `(n_samples, n_features)`.
        name: Identifiant du lot (ex. `"dcgan_seed42"`).
        directory: Repertoire de destination.

    Returns:
        Le chemin du `.npy` ecrit.
    """
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{name}.npy"
    np.save(destination, np.asarray(X_synthetic))
    return destination


def load_synthetic(
    name: str,
    *,
    n_features: int | None = None,
    directory: Path = config.DATA_SYNTHETIC,
) -> np.ndarray:
    """Charge un lot d'echantillons generes en validant sa forme.

    Point d'entree unique des donnees venues du perimetre ML : la validation se
    fait ici, une fois, plutot que dans chaque script consommateur.

    Args:
        name: Identifiant utilise a la sauvegarde.
        n_features: Nombre de colonnes attendu. `None` desactive le controle.
        directory: Repertoire de recherche.

    Returns:
        La matrice chargee.

    Raises:
        FileNotFoundError: Si le lot n'existe pas.
        ValueError: Si la forme ne correspond pas a l'attendu.
    """
    source = directory / f"{name}.npy"
    if not source.exists():
        raise FileNotFoundError(
            f"Aucun lot synthetique a {source}. "
            "Les echantillons generes doivent etre deposes dans "
            f"{directory} au format .npy."
        )

    X = np.load(source)
    if X.ndim != 2:
        raise ValueError(f"{source} doit contenir une matrice 2-D, recu {X.shape}.")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(
            f"{source} a {X.shape[1]} features, {n_features} attendues."
        )
    return X
