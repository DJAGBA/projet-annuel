"""Pipeline tabulaire Credit Card Fraud (mlg-ulb/creditcardfraud).

Le dataset compte ~284 807 transactions dont ~0.17 % de fraudes : c'est ce
desequilibre extreme qui justifie l'augmentation par GAN, et c'est lui qui rend
chaque etape du preprocessing sensible.

**L'ordre des operations est le coeur de ce module.** Il est impose et non
negociable :

1. split train/test **stratifie**, sur donnees reelles uniquement ;
2. `fit` du scaler sur le **train seul**, puis `transform` du test ;
3. isolation de la classe minoritaire **issue du train**.

Inverser 1 et 2 -- c'est-a-dire scaler avant de splitter -- ferait fuiter la
moyenne et l'ecart-type du test dans le train. La fuite est silencieuse : rien
ne plante, les scores montent, et la comparaison DCGAN / WGAN-GP devient
ininterpretable.

Aucun credential n'apparait ici : `kagglehub` et le CLI `kaggle` lisent
eux-memes `~/.kaggle/kaggle.json` ou `KAGGLE_USERNAME` / `KAGGLE_KEY`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data import config
from src.data.splits import (
    assert_disjoint,
    class_balance,
    split_fingerprint,
    stratified_indices,
)

__all__ = [
    "TabularSplit",
    "ensure_creditcard_csv",
    "load_raw",
    "describe_imbalance",
    "build_features",
    "preprocess_tabular",
    "extract_minority",
    "save_processed",
    "load_processed",
]


MANUAL_INSTRUCTIONS = f"""
Le fichier {config.CREDITCARD_CSV_NAME} est introuvable et le telechargement
automatique a echoue.

Option A -- credentials Kaggle (telechargement automatique) :
  1. Creer un token sur https://www.kaggle.com/settings > "Create New Token".
  2. Placer le fichier telecharge dans ~/.kaggle/kaggle.json
     (Windows : %USERPROFILE%\\.kaggle\\kaggle.json)
     OU exporter KAGGLE_USERNAME et KAGGLE_KEY dans l'environnement.
  3. Relancer : python -m src.data.tabular --download

Option B -- telechargement manuel :
  1. Ouvrir https://www.kaggle.com/datasets/{config.CREDITCARD_KAGGLE_ID}
  2. Telecharger l'archive, en extraire {config.CREDITCARD_CSV_NAME}
  3. Deposer le fichier a : {config.CREDITCARD_CSV}

Ne jamais commiter kaggle.json ni ecrire de credential dans le code.
""".strip()


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def _has_kaggle_credentials() -> bool:
    """Indique si des credentials Kaggle sont disponibles.

    Verifie uniquement la **presence** d'un token, jamais son contenu : la
    valeur reste entre les mains de la bibliotheque Kaggle.
    """
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def _locate_csv(directory: Path) -> Path | None:
    """Retrouve `creditcard.csv` dans une arborescence telechargee."""
    candidate = directory / config.CREDITCARD_CSV_NAME
    if candidate.exists():
        return candidate
    return next(iter(directory.rglob(config.CREDITCARD_CSV_NAME)), None)


def _download_with_kagglehub(destination: Path) -> Path:
    """Telecharge via `kagglehub` (gere son propre cache et ses credentials)."""
    import kagglehub

    cached = Path(kagglehub.dataset_download(config.CREDITCARD_KAGGLE_ID))
    source = _locate_csv(cached)
    if source is None:
        raise FileNotFoundError(
            f"{config.CREDITCARD_CSV_NAME} absent du telechargement kagglehub ({cached})."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _download_with_kaggle_cli(destination: Path) -> Path:
    """Telecharge via le CLI `kaggle`, qui lit lui aussi kaggle.json / l'env."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            config.CREDITCARD_KAGGLE_ID,
            "-p",
            str(destination.parent),
            "--unzip",
        ],
        check=True,
        capture_output=True,
    )
    source = _locate_csv(destination.parent)
    if source is None:
        raise FileNotFoundError(
            f"{config.CREDITCARD_CSV_NAME} absent apres extraction du CLI kaggle."
        )
    if source != destination:
        shutil.move(str(source), destination)
    return destination


def ensure_creditcard_csv(
    destination: Path = config.CREDITCARD_CSV,
    *,
    allow_download: bool = True,
) -> Path:
    """Garantit la presence du CSV, en le telechargeant si necessaire.

    Essaie `kagglehub` puis le CLI `kaggle`. Si les deux echouent, leve une
    erreur porteuse d'instructions actionnables plutot que d'une trace obscure.

    Args:
        destination: Emplacement cible du CSV.
        allow_download: Si `False`, se contente de verifier la presence du
            fichier (utile en test pour ne pas declencher de reseau).

    Returns:
        Le chemin du CSV.

    Raises:
        RuntimeError: Si le fichier est absent et n'a pas pu etre telecharge.
            Le message contient la marche a suivre.
    """
    if destination.exists():
        return destination

    if not allow_download:
        raise RuntimeError(MANUAL_INSTRUCTIONS)

    failures: list[str] = []
    if not _has_kaggle_credentials():
        failures.append(
            "aucun credential detecte (~/.kaggle/kaggle.json ou KAGGLE_USERNAME/KAGGLE_KEY)"
        )

    for strategy in (_download_with_kagglehub, _download_with_kaggle_cli):
        try:
            return strategy(destination)
        except Exception as exc:  # noqa: BLE001 - on essaie la strategie suivante
            detail = getattr(exc, "stderr", b"") or b""
            if isinstance(detail, bytes):
                detail = detail.decode(errors="replace")
            failures.append(f"{strategy.__name__}: {type(exc).__name__} {exc} {detail}".strip())

    raise RuntimeError(MANUAL_INSTRUCTIONS + "\n\nEchecs rencontres :\n- " + "\n- ".join(failures))


def load_raw(
    csv_path: Path | None = None,
    *,
    allow_download: bool = True,
) -> pd.DataFrame:
    """Charge le CSV brut et valide son schema.

    Args:
        csv_path: Chemin explicite. `None` utilise `config.CREDITCARD_CSV`.
        allow_download: Autorise le telechargement si le fichier est absent.

    Returns:
        Le `DataFrame` brut, colonnes inchangees.

    Raises:
        ValueError: Si des colonnes attendues manquent.
    """
    path = csv_path or ensure_creditcard_csv(allow_download=allow_download)
    frame = pd.read_csv(path)

    expected = {*config.V_COLUMNS, config.AMOUNT_COL, config.TIME_COL, config.TARGET_COL}
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans {path} : {sorted(missing)}. "
            "Le fichier n'est probablement pas le dataset mlg-ulb/creditcardfraud."
        )
    return frame


# ---------------------------------------------------------------------------
# EDA minimale (elle sert a justifier les choix de preprocessing)
# ---------------------------------------------------------------------------


def describe_imbalance(frame: pd.DataFrame) -> dict[str, float]:
    """Mesure le desequilibre et la dissymetrie d'`Amount`.

    Volontairement limitee a ce qui motive une decision de preprocessing :
    le ratio de fraudes justifie la stratification du split, l'asymetrie
    d'`Amount` justifie le `log1p`. L'analyse metier ne releve pas de ce module.

    Args:
        frame: DataFrame brut.

    Returns:
        Un dict de statistiques scalaires, serialisable en JSON.
    """
    target = frame[config.TARGET_COL]
    amount = frame[config.AMOUNT_COL]
    n_fraud = int((target == config.FRAUD_LABEL).sum())

    return {
        "n_rows": int(len(frame)),
        "n_fraud": n_fraud,
        "n_legit": int(len(frame) - n_fraud),
        "fraud_ratio": float(n_fraud / len(frame)),
        "fraud_percent": float(100.0 * n_fraud / len(frame)),
        "imbalance_ratio": float((len(frame) - n_fraud) / max(n_fraud, 1)),
        "amount_mean": float(amount.mean()),
        "amount_median": float(amount.median()),
        "amount_std": float(amount.std()),
        "amount_max": float(amount.max()),
        "amount_skew": float(amount.skew()),
        "amount_skew_log1p": float(np.log1p(amount).skew()),
    }


# ---------------------------------------------------------------------------
# Construction des features (transformations sans parametre appris)
# ---------------------------------------------------------------------------


def build_features(
    frame: pd.DataFrame,
    *,
    time_strategy: Literal["drop", "cyclical"] = config.TIME_STRATEGY,
    amount_log1p: bool = config.AMOUNT_LOG1P,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Construit la matrice de features et le vecteur cible.

    N'applique que des transformations **sans parametre appris** (drop, log1p,
    encodage trigonometrique). C'est la raison pour laquelle cette etape peut
    legitimement preceder le split : appliquer `log1p` ligne a ligne n'estime
    rien sur la population, donc ne peut rien faire fuiter du test vers le
    train. Toute transformation ajustee (`StandardScaler`) vient apres le split.

    `Time` : le dataset l'exprime en secondes depuis la premiere transaction,
    c'est-a-dire un index d'acquisition. La conserver telle quelle apprendrait
    au GAN un artefact de collecte, et le split aleatoire en brise de toute
    facon l'ordre. Deux strategies :

    * `"drop"` (defaut) : colonne retiree ;
    * `"cyclical"` : convertie en heure de la journee puis encodee en
      (sin, cos), ce qui conserve le rythme circadien sans la derive absolue
      et evite la discontinuite artificielle entre 23h59 et 00h00.

    Args:
        frame: DataFrame brut.
        time_strategy: Sort reserve a la colonne `Time`.
        amount_log1p: Applique `log1p` a `Amount` avant standardisation.

    Returns:
        Le couple `(features, y)`.

    Raises:
        ValueError: Si `time_strategy` est inconnue.
    """
    features = frame[list(config.V_COLUMNS)].copy()

    amount = frame[config.AMOUNT_COL].astype(float)
    # `Amount` a une queue lourde : sans log1p, le scaler est domine par
    # quelques transactions extremes et ecrase la variabilite du gros du jeu.
    features[config.AMOUNT_COL] = np.log1p(amount) if amount_log1p else amount

    if time_strategy == "cyclical":
        seconds_of_day = frame[config.TIME_COL].astype(float) % config.SECONDS_PER_DAY
        angle = 2.0 * np.pi * seconds_of_day / config.SECONDS_PER_DAY
        features["time_sin"] = np.sin(angle)
        features["time_cos"] = np.cos(angle)
    elif time_strategy != "drop":
        raise ValueError(
            f"time_strategy inconnue : {time_strategy!r}. Attendu 'drop' ou 'cyclical'."
        )

    y = frame[config.TARGET_COL].to_numpy(dtype=np.int64)
    return features, y


# ---------------------------------------------------------------------------
# Resultat du pipeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TabularSplit:
    """Sortie complete du preprocessing tabulaire.

    Attributes:
        X_train: Features d'entrainement scalees.
        X_test: Features de test scalees (transformees, jamais ajustees).
        y_train: Etiquettes d'entrainement.
        y_test: Etiquettes de test.
        scaler: `StandardScaler` ajuste sur le train seul. Conserve car son
            `inverse_transform` est necessaire pour ramener les echantillons
            generes par le GAN dans l'espace des grandeurs reelles.
        feature_names: Noms des colonnes, dans l'ordre des matrices.
        train_index: Indices des lignes du DataFrame source formant le train.
        test_index: Indices des lignes du DataFrame source formant le test.
        seed: Graine ayant produit le split.
        time_strategy: Sort applique a la colonne `Time`.
        amount_log1p: `log1p` applique a `Amount`.
    """

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    feature_names: tuple[str, ...]
    train_index: np.ndarray
    test_index: np.ndarray
    seed: int
    time_strategy: str
    amount_log1p: bool

    @property
    def minority_train(self) -> np.ndarray:
        """Fraudes **du train** : le seul materiau licite pour entrainer le GAN."""
        return extract_minority(self.X_train, self.y_train)

    @property
    def fingerprint(self) -> str:
        """Empreinte du split, pour verifier deux runs sans comparer les donnees."""
        return split_fingerprint(self.train_index, self.test_index)

    def summary(self) -> dict[str, object]:
        """Resume serialisable en JSON, ecrit a cote des artefacts."""
        return {
            "seed": self.seed,
            "time_strategy": self.time_strategy,
            "amount_log1p": self.amount_log1p,
            "n_features": len(self.feature_names),
            "feature_names": list(self.feature_names),
            "n_train": int(self.X_train.shape[0]),
            "n_test": int(self.X_test.shape[0]),
            "n_minority_train": int(self.minority_train.shape[0]),
            "balance_train": class_balance(self.y_train),
            "balance_test": class_balance(self.y_test),
            "fingerprint": self.fingerprint,
        }


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------


def preprocess_tabular(
    seed: int = config.SEED,
    *,
    test_size: float = config.TEST_SIZE,
    time_strategy: Literal["drop", "cyclical"] = config.TIME_STRATEGY,
    amount_log1p: bool = config.AMOUNT_LOG1P,
    frame: pd.DataFrame | None = None,
    csv_path: Path | None = None,
    allow_download: bool = True,
) -> TabularSplit:
    """Execute le pipeline tabulaire dans l'ordre impose contre le leakage.

    Args:
        seed: Graine du split.
        test_size: Proportion du jeu de test.
        time_strategy: Sort reserve a `Time`.
        amount_log1p: Applique `log1p` a `Amount`.
        frame: DataFrame deja charge (utile en test). `None` declenche
            `load_raw`.
        csv_path: Chemin explicite du CSV.
        allow_download: Autorise le telechargement Kaggle si besoin.

    Returns:
        Un `TabularSplit` complet.
    """
    source = load_raw(csv_path, allow_download=allow_download) if frame is None else frame
    features, y = build_features(
        source, time_strategy=time_strategy, amount_log1p=amount_log1p
    )
    values = features.to_numpy(dtype=np.float64)

    # --- 1. Split stratifie D'ABORD, sur donnees reelles -------------------
    train_index, test_index = stratified_indices(y, test_size=test_size, seed=seed)
    assert_disjoint(train_index, test_index)

    X_train_raw, X_test_raw = values[train_index], values[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # --- 2. Scaler ajuste sur le TRAIN SEUL, puis applique au test ---------
    # `fit_transform` sur le train, `transform` (jamais `fit`) sur le test :
    # le test ne contribue a aucune statistique.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # --- 3. Isolation de la classe minoritaire ISSUE DU TRAIN --------------
    # (expose via la propriete `minority_train`, calculee a la demande)

    return TabularSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
        feature_names=tuple(features.columns),
        train_index=train_index,
        test_index=test_index,
        seed=seed,
        time_strategy=time_strategy,
        amount_log1p=amount_log1p,
    )


def extract_minority(
    X: np.ndarray,
    y: np.ndarray,
    label: int = config.FRAUD_LABEL,
) -> np.ndarray:
    """Extrait les lignes de la classe minoritaire.

    Args:
        X: Matrice de features.
        y: Etiquettes alignees sur `X`.
        label: Etiquette a isoler.

    Returns:
        Le sous-ensemble des lignes de `X` portant `label`.

    Raises:
        ValueError: Si `X` et `y` ne sont pas alignes.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X ({X.shape[0]}) et y ({y.shape[0]}) ne sont pas alignes.")
    return X[y == label]


# ---------------------------------------------------------------------------
# Persistance
# ---------------------------------------------------------------------------


def _artifact_paths(directory: Path) -> dict[str, Path]:
    """Chemins des artefacts produits par `save_processed`."""
    prefix = config.PROCESSED_PREFIX
    return {
        "X_train": directory / f"{prefix}_X_train.npy",
        "X_test": directory / f"{prefix}_X_test.npy",
        "y_train": directory / f"{prefix}_y_train.npy",
        "y_test": directory / f"{prefix}_y_test.npy",
        "X_minority_train": directory / f"{prefix}_X_minority_train.npy",
        "train_index": directory / f"{prefix}_train_index.npy",
        "test_index": directory / f"{prefix}_test_index.npy",
        "scaler": directory / config.SCALER_FILENAME,
        "metadata": directory / config.PROCESSED_METADATA,
    }


def save_processed(
    split: TabularSplit,
    directory: Path = config.DATA_PROCESSED,
) -> dict[str, Path]:
    """Persiste arrays, scaler ajuste et metadonnees.

    Le scaler est sauvegarde avec les donnees et non recalcule a la volee :
    c'est lui qui porte la garantie "ajuste sur le train seul". Le regenerer
    ailleurs, c'est risquer de le refitter sur autre chose.

    Args:
        split: Sortie de `preprocess_tabular`.
        directory: Repertoire de destination.

    Returns:
        Le dict `nom logique -> chemin ecrit`.
    """
    directory.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(directory)

    np.save(paths["X_train"], split.X_train)
    np.save(paths["X_test"], split.X_test)
    np.save(paths["y_train"], split.y_train)
    np.save(paths["y_test"], split.y_test)
    np.save(paths["X_minority_train"], split.minority_train)
    np.save(paths["train_index"], split.train_index)
    np.save(paths["test_index"], split.test_index)
    joblib.dump(split.scaler, paths["scaler"])
    paths["metadata"].write_text(
        json.dumps(split.summary(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return paths


def load_processed(directory: Path = config.DATA_PROCESSED) -> TabularSplit:
    """Recharge un `TabularSplit` persiste.

    Args:
        directory: Repertoire contenant les artefacts.

    Returns:
        Le `TabularSplit` reconstitue.

    Raises:
        FileNotFoundError: Si un artefact attendu manque.
    """
    paths = _artifact_paths(directory)
    missing = [str(path) for key, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Artefacts tabulaires manquants : "
            + ", ".join(missing)
            + ". Lancer `python -m src.data.tabular`."
        )

    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    return TabularSplit(
        X_train=np.load(paths["X_train"]),
        X_test=np.load(paths["X_test"]),
        y_train=np.load(paths["y_train"]),
        y_test=np.load(paths["y_test"]),
        scaler=joblib.load(paths["scaler"]),
        feature_names=tuple(metadata["feature_names"]),
        train_index=np.load(paths["train_index"]),
        test_index=np.load(paths["test_index"]),
        seed=int(metadata["seed"]),
        time_strategy=str(metadata["time_strategy"]),
        amount_log1p=bool(metadata["amount_log1p"]),
    )


def main() -> None:
    """CLI : `python -m src.data.tabular [--download] [--seed N] [--eda]`."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--download",
        action="store_true",
        help="Telecharge le CSV depuis Kaggle puis s'arrete.",
    )
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--test-size", type=float, default=config.TEST_SIZE)
    parser.add_argument(
        "--time-strategy", choices=("drop", "cyclical"), default=config.TIME_STRATEGY
    )
    parser.add_argument("--eda", action="store_true", help="Affiche l'EDA rapide.")
    args = parser.parse_args()

    try:
        _run(args)
    except RuntimeError as exc:
        # Dataset absent sans credential : condition attendue et actionnable.
        # L'utilisateur a besoin de la marche a suivre, pas d'une traceback.
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None


def _run(args) -> None:
    """Corps du CLI, isole pour que `main` ne gere que la presentation."""
    if args.download:
        print(f"[ok] CSV disponible : {ensure_creditcard_csv()}")
        return

    frame = load_raw()
    if args.eda:
        for key, value in describe_imbalance(frame).items():
            print(f"  {key:>20} : {value}")

    split = preprocess_tabular(
        args.seed,
        test_size=args.test_size,
        time_strategy=args.time_strategy,
        frame=frame,
    )
    paths = save_processed(split)

    summary = split.summary()
    print(
        f"[ok] train={summary['n_train']} test={summary['n_test']} "
        f"fraudes_train={summary['n_minority_train']} "
        f"features={summary['n_features']} "
        f"fingerprint={summary['fingerprint'][:12]}..."
    )
    print(f"[ok] artefacts -> {paths['metadata'].parent}")


if __name__ == "__main__":
    main()
