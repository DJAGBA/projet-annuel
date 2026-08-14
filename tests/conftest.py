"""Fixtures partagees et helpers de skip conditionnel.

Principe : `pytest` doit passer integralement sur une machine vierge, sans
telechargement prealable. Les tests qui dependent d'un dataset absent sont
skippes avec un message actionnable, jamais en echec.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data import config
from src.data.utils import ensure_dirs, set_seed


@pytest.fixture(autouse=True)
def _deterministic_environment() -> None:
    """Reseed avant chaque test : aucun test ne depend de l'ordre d'execution."""
    set_seed(config.SEED)


@pytest.fixture(scope="session", autouse=True)
def _project_dirs() -> None:
    """Garantit l'existence de l'arborescence de travail."""
    ensure_dirs()


@pytest.fixture(scope="session")
def creditcard_csv() -> Path:
    """Chemin du CSV Credit Card Fraud, ou skip s'il n'a pas ete telecharge."""
    if not config.CREDITCARD_CSV.exists():
        pytest.skip(
            f"{config.CREDITCARD_CSV} absent. "
            "Lancer `python -m src.data.tabular --download` (cf. README)."
        )
    return config.CREDITCARD_CSV
