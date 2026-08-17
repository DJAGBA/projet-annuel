"""Fixtures partagees et helpers de skip conditionnel.

Principe : `pytest` doit passer integralement sur une machine vierge, sans
telechargement prealable. Les tests qui dependent d'un dataset absent sont
skippes avec un message actionnable, jamais en echec.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from torch.utils.data import DataLoader

from src.data import config
from src.data.images import get_image_dataloader
from src.data.utils import ensure_dirs, set_seed


@pytest.fixture(autouse=True)
def _deterministic_environment() -> None:
    """Reseed avant chaque test : aucun test ne depend de l'ordre d'execution."""
    set_seed(config.SEED)


@pytest.fixture(scope="session", autouse=True)
def _project_dirs() -> None:
    """Garantit l'existence de l'arborescence de travail."""
    ensure_dirs()


@pytest.fixture
def image_loaders() -> Callable[..., tuple[DataLoader, DataLoader]]:
    """Fabrique de DataLoaders images, ou skip si le dataset n'est pas sur disque.

    Volontairement **sans cache** : chaque appel reconstruit des `DataLoader`
    neufs. Un `DataLoader` partage verrait son `torch.Generator` avancer d'une
    iteration a l'autre (comportement normal entre deux epoques), ce qui rendrait
    les tests de reproductibilite faussement rouges.
    """

    def _build(
        dataset_name: str,
        *,
        seed: int = config.SEED,
        batch_size: int = 16,
        **kwargs: object,
    ) -> tuple[DataLoader, DataLoader]:
        try:
            return get_image_dataloader(
                dataset_name,
                batch_size=batch_size,
                seed=seed,
                num_workers=0,
                download=False,
                **kwargs,  # type: ignore[arg-type]
            )
        except (RuntimeError, FileNotFoundError) as exc:
            pytest.skip(
                f"Dataset {dataset_name!r} absent de {config.DATA_RAW}. "
                f"Lancer `python -m src.data.images --dataset {dataset_name}`. "
                f"({type(exc).__name__})"
            )

    return _build


@pytest.fixture(scope="session")
def creditcard_csv() -> Path:
    """Chemin du CSV Credit Card Fraud, ou skip s'il n'a pas ete telecharge."""
    if not config.CREDITCARD_CSV.exists():
        pytest.skip(
            f"{config.CREDITCARD_CSV} absent. "
            "Lancer `python -m src.data.tabular --download` (cf. README)."
        )
    return config.CREDITCARD_CSV
