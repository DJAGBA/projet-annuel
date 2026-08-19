"""Phase 5 — verification directe des criteres d'acceptation du projet.

Les autres fichiers testent des garanties internes ; celui-ci teste le
**contrat livre** : surface publique utilisable, absence de secret dans tout
`src/`, et artefacts exclus du versionnement. Ces proprietes se degradent
silencieusement au fil des commits, d'ou leur mise sous test.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import config

SOURCE_FILES = sorted((config.PROJECT_ROOT / "src").rglob("*.py"))


# ---------------------------------------------------------------------------
# Critere : les entrees publiques sont claires et fonctionnelles
# ---------------------------------------------------------------------------


def test_public_entry_points_are_importable() -> None:
    """Les fonctions citees dans le README doivent exister et etre appelables."""
    from src.data.augment import inject_synthetic
    from src.data.fid_stats import get_or_compute_fid_statistics
    from src.data.images import get_image_dataloader
    from src.data.tabular import load_processed, preprocess_tabular, save_processed

    for entry in (
        get_image_dataloader,
        preprocess_tabular,
        save_processed,
        load_processed,
        inject_synthetic,
        get_or_compute_fid_statistics,
    ):
        assert callable(entry)
        assert entry.__doc__, f"{entry.__name__} doit etre documentee."


@pytest.mark.requires_download
def test_an_image_dataloader_is_generable_in_one_call(image_loaders) -> None:
    """Critere : "un DataLoader image est generable via une fonction publique"."""
    from src.data.images import sanity_check_batch

    train_loader, _ = image_loaders("cifar10")
    images, _ = next(iter(train_loader))
    sanity_check_batch(images, expected_channels=3, expected_size=config.IMAGE_SIZE)


def test_the_tabular_set_is_generable_and_persistable(
    synthetic_creditcard: pd.DataFrame, tmp_path
) -> None:
    """Critere : "le jeu tabulaire processe est generable via une fonction publique"."""
    from src.data.tabular import load_processed, preprocess_tabular, save_processed

    split = preprocess_tabular(frame=synthetic_creditcard)
    save_processed(split, directory=tmp_path)
    reloaded = load_processed(directory=tmp_path)

    np.testing.assert_array_equal(reloaded.X_train, split.X_train)
    assert reloaded.minority_train.shape[0] > 0


def test_every_public_module_exposes_an_explicit_surface() -> None:
    """`__all__` documente ce qui est stable ; son absence rend le contrat flou."""
    import importlib

    for name in ("images", "tabular", "splits", "augment", "fid_stats", "utils"):
        module = importlib.import_module(f"src.data.{name}")
        assert getattr(module, "__all__", None), f"src/data/{name}.py doit definir __all__."


# ---------------------------------------------------------------------------
# Critere : aucun secret dans le code
# ---------------------------------------------------------------------------


SECRET_ASSIGNMENT = re.compile(
    r"""(KAGGLE_KEY|KAGGLE_USERNAME|api_key|apikey|password|passwd|secret|token)"""
    r"""\s*[:=]\s*["'][^"']+["']""",
    re.IGNORECASE,
)


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: p.name)
def test_no_source_file_assigns_a_credential(path: Path) -> None:
    match = SECRET_ASSIGNMENT.search(path.read_text(encoding="utf-8"))
    assert match is None, f"Credential potentiellement en dur dans {path}: {match.group(0)!r}"


def test_credentials_are_only_ever_read_from_the_environment() -> None:
    """Les credentials Kaggle doivent venir de l'env ou de kaggle.json, jamais du code."""
    source = (config.PROJECT_ROOT / "src" / "data" / "tabular.py").read_text(
        encoding="utf-8"
    )
    assert "os.environ.get(\"KAGGLE_USERNAME\")" in source
    assert ".kaggle" in source and "kaggle.json" in source


# ---------------------------------------------------------------------------
# Critere : datasets et artefacts hors git
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gitignore() -> str:
    path = config.PROJECT_ROOT / ".gitignore"
    if not path.exists():
        pytest.fail("Un .gitignore est requis : les datasets ne doivent pas etre versionnes.")
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "pattern",
    ["data/raw/", "data/processed/", "data/synthetic/", "artifacts/", "kaggle.json"],
)
def test_generated_content_is_excluded_from_git(pattern: str, gitignore: str) -> None:
    assert pattern in gitignore, f"{pattern} doit figurer dans .gitignore."


def test_data_directories_survive_cloning() -> None:
    """Les `.gitkeep` preservent l'arborescence malgre l'exclusion du contenu."""
    for directory in (config.DATA_RAW, config.DATA_PROCESSED, config.DATA_SYNTHETIC):
        assert (directory / ".gitkeep").exists(), f"{directory}/.gitkeep manquant."


# ---------------------------------------------------------------------------
# Critere : le README documente credentials et etapes
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def readme() -> str:
    candidates = [config.PROJECT_ROOT / "README.md", config.PROJECT_ROOT / "readme.md"]
    for path in candidates:
        if path.exists():
            return path.read_text(encoding="utf-8")
    pytest.fail("Aucun README trouve a la racine du depot.")


@pytest.mark.parametrize(
    "topic",
    ["kaggle.json", "KAGGLE_USERNAME", "KAGGLE_KEY", config.CREDITCARD_KAGGLE_ID],
)
def test_readme_documents_kaggle_credentials(topic: str, readme: str) -> None:
    assert topic in readme, f"Le README doit expliquer {topic}."


@pytest.mark.parametrize(
    "command",
    [
        "pip install -r requirements.txt",
        "python -m src.data.images",
        "python -m src.data.tabular",
        "python -m src.data.fid_stats",
        "pytest",
    ],
)
def test_readme_documents_how_to_run_each_step(command: str, readme: str) -> None:
    assert command in readme, f"Le README doit documenter la commande : {command}"
