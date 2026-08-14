"""Utilitaires transverses : reproductibilite, device, arborescence.

Le point critique de ce module est `set_seed` : sans lui, deux runs du meme
pipeline produisent des splits differents et la comparaison DCGAN / WGAN-GP
n'a plus de valeur scientifique.
"""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path

import numpy as np
import torch

from src.data import config

__all__ = [
    "set_seed",
    "seed_worker",
    "make_generator",
    "ensure_dirs",
    "get_device",
    "resolve_pin_memory",
    "hash_array",
]


def set_seed(seed: int = config.SEED, *, strict: bool | None = None) -> None:
    """Fixe toutes les sources d'alea et active le mode deterministe.

    Couvre `random`, `numpy`, `torch` (CPU **et** CUDA, tous devices), le hash
    seed de l'interpreteur, ainsi que les backends cuDNN / cuBLAS.

    Args:
        seed: Graine a appliquer. Defaut : `config.SEED`.
        strict: Si `True`, `torch.use_deterministic_algorithms` leve une
            exception sur toute operation non deterministe. Si `False`, elle
            se contente d'un warning. `None` reprend `config.STRICT_DETERMINISM`.

            Le mode non strict est le defaut : plusieurs backwards utilises par
            les generateurs convolutionnels n'ont pas d'implementation
            deterministe CUDA et feraient echouer l'entrainement.

    Note:
        `CUBLAS_WORKSPACE_CONFIG` doit etre positionnee *avant* la premiere
        initialisation de cuBLAS. Appeler `set_seed()` en tout debut de script
        (avant toute operation matricielle GPU) garantit son effet.
    """
    if strict is None:
        strict = config.STRICT_DETERMINISM

    os.environ["PYTHONHASHSEED"] = str(seed)
    # Requis par cuBLAS pour un GEMM deterministe sur CUDA >= 10.2.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # l'autotuner rend les runs variables
    torch.use_deterministic_algorithms(True, warn_only=not strict)


def seed_worker(worker_id: int) -> None:
    """Re-seed un worker de `DataLoader`.

    A passer en `worker_init_fn`. Sans cela, chaque process worker herite d'un
    etat numpy/random non controle : le `set_seed` du process principal ne
    suffit pas des que `num_workers > 0`.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = config.SEED) -> torch.Generator:
    """Cree un `torch.Generator` dedie au shuffle d'un `DataLoader`.

    Isoler le generateur du RNG global rend l'ordre des batchs independant du
    nombre d'appels aleatoires faits ailleurs (init des poids, dropout...).
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def ensure_dirs(*extra: Path) -> None:
    """Cree les repertoires de travail du projet s'ils n'existent pas."""
    for directory in (*config.MANAGED_DIRS, *extra):
        directory.mkdir(parents=True, exist_ok=True)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Retourne le device a utiliser (`cuda` si disponible et souhaite)."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_pin_memory() -> bool:
    """`pin_memory` n'a de sens que si un GPU consomme les batchs."""
    return torch.cuda.is_available()


def hash_array(array: np.ndarray) -> str:
    """Empreinte SHA-256 stable d'un tableau numpy.

    Sert de signature de split / de ligne : deux runs a seed identique doivent
    produire exactement le meme hash (cf. `tests/test_reproducibility.py`).

    Args:
        array: Tableau a hasher. Il est rendu contigu et son dtype/shape sont
            integres a l'empreinte pour eviter les collisions entre tableaux
            de meme contenu binaire mais de forme differente.

    Returns:
        L'empreinte hexadecimale.
    """
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(str(contiguous.shape).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()
