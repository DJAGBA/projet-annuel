"""Phase 3 — reproductibilite : meme seed, memes donnees, d'un run a l'autre.

Sans cette garantie, un ecart de FID entre DCGAN et WGAN-GP melange deux
causes : la difference d'architecture et le hasard du decoupage. La comparaison
n'est alors plus interpretable.

Chaque garantie est testee avec son garde-fou anti faux-positif : un test de
reproductibilite qui ne saurait pas echouer ne prouverait rien.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.data import config
from src.data.augment import inject_synthetic
from src.data.images import batch_order_fingerprint
from src.data.splits import (
    assert_same_split,
    split_fingerprint,
    stratified_indices,
    verify_split_reproducibility,
)
from src.data.tabular import preprocess_tabular, verify_preprocessing_reproducibility
from src.data.utils import hash_array, set_seed


# ---------------------------------------------------------------------------
# Pipeline tabulaire complet
# ---------------------------------------------------------------------------


def test_pipeline_is_bit_identical_across_runs(
    synthetic_creditcard: pd.DataFrame,
) -> None:
    """Indices, matrices scalees et scaler doivent coincider exactement."""
    fingerprint = verify_preprocessing_reproducibility(
        synthetic_creditcard, seed=config.SEED, n_runs=3
    )
    assert len(fingerprint) == 64


def test_pipeline_diverges_when_the_seed_changes(
    synthetic_creditcard: pd.DataFrame,
) -> None:
    """Garde-fou anti faux-positif du test precedent."""
    first = preprocess_tabular(config.SEED, frame=synthetic_creditcard)
    other = preprocess_tabular(config.SEED + 1, frame=synthetic_creditcard)
    assert first.fingerprint != other.fingerprint


def test_scaled_values_are_identical_across_runs(
    synthetic_creditcard: pd.DataFrame,
) -> None:
    first = preprocess_tabular(frame=synthetic_creditcard)
    second = preprocess_tabular(frame=synthetic_creditcard)

    assert hash_array(first.X_train) == hash_array(second.X_train)
    assert hash_array(first.X_test) == hash_array(second.X_test)
    np.testing.assert_array_equal(first.scaler.mean_, second.scaler.mean_)


def test_pipeline_is_immune_to_ambient_rng_state(
    synthetic_creditcard: pd.DataFrame,
) -> None:
    """Un tirage aleatoire parasite ne doit pas deplacer le split."""
    reference = preprocess_tabular(frame=synthetic_creditcard)

    set_seed(config.SEED + 99)
    np.random.rand(10_000)
    torch.rand(1_000)
    after_noise = preprocess_tabular(frame=synthetic_creditcard)

    assert reference.fingerprint == after_noise.fingerprint
    assert hash_array(reference.X_train) == hash_array(after_noise.X_train)


@pytest.mark.parametrize("seed", config.MULTI_SEEDS)
def test_each_configured_seed_is_reproducible(
    seed: int, synthetic_creditcard: pd.DataFrame
) -> None:
    """Les runs multi-seed doivent etre rejouables un par un."""
    fingerprint = verify_preprocessing_reproducibility(
        synthetic_creditcard, seed=seed, n_runs=2
    )
    assert len(fingerprint) == 64


def test_configured_seeds_produce_distinct_splits(
    synthetic_creditcard: pd.DataFrame,
) -> None:
    """Sinon le protocole multi-seed ne mesurerait aucune variabilite."""
    fingerprints = {
        preprocess_tabular(seed, frame=synthetic_creditcard).fingerprint
        for seed in config.MULTI_SEEDS
    }
    assert len(fingerprints) == len(config.MULTI_SEEDS)


# ---------------------------------------------------------------------------
# Verification active des splits
# ---------------------------------------------------------------------------


def test_verify_split_reproducibility_returns_a_stable_fingerprint() -> None:
    y = np.zeros(500, dtype=np.int64)
    y[:9] = 1
    fingerprint = verify_split_reproducibility(y, seed=config.SEED, n_runs=4)
    assert fingerprint == split_fingerprint(*stratified_indices(y, seed=config.SEED))


def test_verify_split_reproducibility_requires_at_least_two_runs() -> None:
    with pytest.raises(ValueError, match="au moins 2"):
        verify_split_reproducibility(np.array([0, 1, 0, 1]), n_runs=1)


def test_assert_same_split_rejects_divergent_fingerprints() -> None:
    with pytest.raises(AssertionError, match="pas vu le meme split"):
        assert_same_split("a" * 64, "b" * 64, context="DCGAN vs WGAN-GP")


def test_assert_same_split_message_names_the_context() -> None:
    with pytest.raises(AssertionError, match="cifar10"):
        assert_same_split("a" * 64, "b" * 64, context="cifar10")


# ---------------------------------------------------------------------------
# Injection synthetique
# ---------------------------------------------------------------------------


def test_injection_is_reproducible_at_fixed_seed(
    synthetic_creditcard: pd.DataFrame,
) -> None:
    """Le melange post-injection doit lui aussi etre deterministe."""
    split = preprocess_tabular(frame=synthetic_creditcard)
    generated = np.random.default_rng(7).normal(size=(40, split.X_train.shape[1]))

    first = inject_synthetic(split.X_train, split.y_train, generated, seed=config.SEED)
    second = inject_synthetic(split.X_train, split.y_train, generated, seed=config.SEED)

    assert hash_array(first.X) == hash_array(second.X)
    np.testing.assert_array_equal(first.is_synthetic, second.is_synthetic)


def test_injection_shuffle_changes_with_the_seed(
    synthetic_creditcard: pd.DataFrame,
) -> None:
    """Garde-fou anti faux-positif du test precedent."""
    split = preprocess_tabular(frame=synthetic_creditcard)
    generated = np.random.default_rng(7).normal(size=(40, split.X_train.shape[1]))

    first = inject_synthetic(split.X_train, split.y_train, generated, seed=config.SEED)
    other = inject_synthetic(
        split.X_train, split.y_train, generated, seed=config.SEED + 1
    )
    assert hash_array(first.X) != hash_array(other.X)


# ---------------------------------------------------------------------------
# Pipeline images
# ---------------------------------------------------------------------------


@pytest.mark.requires_download
def test_two_runs_consume_the_same_image_sequence(image_loaders) -> None:
    """C'est cette egalite qui autorise a comparer DCGAN et WGAN-GP."""
    first, _ = image_loaders("fashion_mnist", seed=config.SEED)
    second, _ = image_loaders("fashion_mnist", seed=config.SEED)
    assert_same_split(
        batch_order_fingerprint(first),
        batch_order_fingerprint(second),
        context="fashion_mnist",
    )


@pytest.mark.requires_download
def test_image_sequence_changes_with_the_seed(image_loaders) -> None:
    """Garde-fou anti faux-positif du test precedent."""
    first, _ = image_loaders("fashion_mnist", seed=config.SEED)
    other, _ = image_loaders("fashion_mnist", seed=config.SEED + 1)
    assert batch_order_fingerprint(first) != batch_order_fingerprint(other)


@pytest.mark.requires_download
def test_evaluation_set_is_identical_regardless_of_the_training_seed(
    image_loaders,
) -> None:
    """Le test n'est pas melange : deux seeds doivent l'evaluer a l'identique."""
    _, first = image_loaders("cifar10", seed=config.SEED)
    _, other = image_loaders("cifar10", seed=config.SEED + 1)
    assert batch_order_fingerprint(first) == batch_order_fingerprint(other)
