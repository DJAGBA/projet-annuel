"""Splits deterministes, reutilisables et verifiables.

Ce module ne manipule que des **indices**, jamais les donnees elles-memes. Deux
consequences utiles : le meme split peut etre rejoue sur des versions
differentes des features (brutes, scalees, augmentees) sans risque de
desynchronisation, et une empreinte compacte suffit a prouver que deux runs ont
bien vu la meme partition.

C'est la brique qui rend la comparaison DCGAN / WGAN-GP legitime : si les deux
modeles ne sont pas evalues sur exactement le meme test, l'ecart de FID mesure
autant le hasard du split que la difference d'architecture.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.data import config
from src.data.utils import hash_array

__all__ = [
    "stratified_indices",
    "split_fingerprint",
    "assert_disjoint",
    "class_balance",
    "save_split_indices",
    "load_split_indices",
]


def stratified_indices(
    y: np.ndarray,
    test_size: float = config.TEST_SIZE,
    seed: int = config.SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcule les indices d'un split train/test stratifie.

    La stratification est indispensable ici : avec ~0.17 % de fraudes, un split
    purement aleatoire ferait varier le nombre de fraudes du test d'un run a
    l'autre, et pourrait meme en produire un sans aucune fraude.

    `random_state=seed` est passe explicitement a scikit-learn plutot que de
    s'appuyer sur l'etat global de numpy : le split est ainsi reproductible
    meme si un appel aleatoire non controle a eu lieu avant.

    Args:
        y: Vecteur d'etiquettes servant de variable de stratification.
        test_size: Proportion du jeu de test.
        seed: Graine du tirage.

    Returns:
        Le couple `(train_index, test_index)`.

    Raises:
        ValueError: Si `y` est vide ou si une classe est trop rare pour etre
            stratifiee (message remonte par scikit-learn).
    """
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"y doit etre 1-D, recu une forme {y.shape}.")
    if y.size == 0:
        raise ValueError("y est vide : aucun split possible.")

    train_index, test_index = train_test_split(
        np.arange(y.size),
        test_size=test_size,
        random_state=seed,
        stratify=y,
        shuffle=True,
    )
    return np.asarray(train_index), np.asarray(test_index)


def split_fingerprint(*index_arrays: np.ndarray) -> str:
    """Empreinte compacte d'un ensemble d'indices.

    Deux runs a seed identique doivent produire la meme empreinte ; c'est la
    verification exploitee par `tests/test_reproducibility.py` et le champ
    stocke dans les metadonnees des artefacts.

    Args:
        *index_arrays: Tableaux d'indices, dans un ordre significatif.

    Returns:
        L'empreinte hexadecimale de la sequence.
    """
    digest = hashlib.sha256()
    for array in index_arrays:
        digest.update(hash_array(np.asarray(array)).encode())
    return digest.hexdigest()


def assert_disjoint(train_index: np.ndarray, test_index: np.ndarray) -> None:
    """Verifie qu'aucun indice n'appartient a la fois au train et au test.

    Args:
        train_index: Indices d'entrainement.
        test_index: Indices de test.

    Raises:
        AssertionError: Si l'intersection est non vide ou si un tableau
            contient des doublons.
    """
    train_index = np.asarray(train_index)
    test_index = np.asarray(test_index)

    overlap = np.intersect1d(train_index, test_index)
    assert overlap.size == 0, (
        f"Fuite : {overlap.size} indice(s) present(s) dans le train ET le test "
        f"(ex. {overlap[:5].tolist()})."
    )
    assert np.unique(train_index).size == train_index.size, "Doublons dans le train."
    assert np.unique(test_index).size == test_index.size, "Doublons dans le test."


def class_balance(y: np.ndarray) -> dict[int, float]:
    """Proportion de chaque classe.

    Args:
        y: Vecteur d'etiquettes.

    Returns:
        Un dict `etiquette -> proportion`, trie par etiquette.
    """
    y = np.asarray(y)
    labels, counts = np.unique(y, return_counts=True)
    return {int(label): float(count / y.size) for label, count in zip(labels, counts)}


def save_split_indices(
    name: str,
    train_index: np.ndarray,
    test_index: np.ndarray,
    *,
    seed: int = config.SEED,
    directory: Path = config.SPLITS_DIR,
) -> Path:
    """Persiste un split pour pouvoir le rejouer a l'identique.

    Args:
        name: Identifiant du split (sert de nom de fichier).
        train_index: Indices d'entrainement.
        test_index: Indices de test.
        seed: Graine ayant produit le split, conservee pour tracabilite.
        directory: Repertoire de destination.

    Returns:
        Le chemin du `.npz` ecrit.
    """
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{name}_split.npz"
    np.savez(
        destination,
        train_index=np.asarray(train_index),
        test_index=np.asarray(test_index),
        seed=np.asarray(seed),
        fingerprint=np.asarray(split_fingerprint(train_index, test_index)),
    )
    return destination


def load_split_indices(
    name: str,
    *,
    directory: Path = config.SPLITS_DIR,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Recharge un split persiste et verifie son integrite.

    Args:
        name: Identifiant utilise a la sauvegarde.
        directory: Repertoire de recherche.

    Returns:
        Le triplet `(train_index, test_index, seed)`.

    Raises:
        FileNotFoundError: Si le split n'a pas ete sauvegarde.
        AssertionError: Si l'empreinte ne correspond plus au contenu, signe
            d'un fichier corrompu ou edite.
    """
    source = directory / f"{name}_split.npz"
    if not source.exists():
        raise FileNotFoundError(f"Aucun split persiste a {source}.")

    with np.load(source, allow_pickle=False) as payload:
        train_index = payload["train_index"]
        test_index = payload["test_index"]
        seed = int(payload["seed"])
        expected = str(payload["fingerprint"])

    actual = split_fingerprint(train_index, test_index)
    assert actual == expected, (
        f"Empreinte du split {name!r} incoherente : attendue {expected}, calculee {actual}."
    )
    return train_index, test_index, seed
