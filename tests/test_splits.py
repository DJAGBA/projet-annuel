"""Phase 2 — primitives de split : stratification, disjonction, empreinte."""

from __future__ import annotations

import numpy as np
import pytest

from src.data import config
from src.data.splits import (
    assert_disjoint,
    class_balance,
    load_split_indices,
    save_split_indices,
    split_fingerprint,
    stratified_indices,
)


@pytest.fixture
def imbalanced_labels() -> np.ndarray:
    """1 000 etiquettes dont 1.7 % de positifs."""
    y = np.zeros(1_000, dtype=np.int64)
    y[:17] = 1
    return y


# ---------------------------------------------------------------------------
# Partition
# ---------------------------------------------------------------------------


def test_split_covers_every_row_exactly_once(imbalanced_labels: np.ndarray) -> None:
    train, test = stratified_indices(imbalanced_labels)
    assert_disjoint(train, test)
    recovered = np.union1d(train, test)
    np.testing.assert_array_equal(recovered, np.arange(imbalanced_labels.size))


def test_test_size_is_respected(imbalanced_labels: np.ndarray) -> None:
    train, test = stratified_indices(imbalanced_labels, test_size=0.2)
    assert test.size == pytest.approx(0.2 * imbalanced_labels.size, abs=1)
    assert train.size + test.size == imbalanced_labels.size


def test_split_preserves_class_proportions(imbalanced_labels: np.ndarray) -> None:
    """Sans stratification, un test pourrait ne contenir aucun positif."""
    train, test = stratified_indices(imbalanced_labels)
    reference = class_balance(imbalanced_labels)

    for subset in (train, test):
        balance = class_balance(imbalanced_labels[subset])
        assert set(balance) == set(reference)
        # Une proportion sur `subset.size` lignes ne peut pas etre plus fine
        # qu'une ligne : c'est la resolution du split, pas une derive.
        tolerance = 1.0 / subset.size
        for label, proportion in reference.items():
            assert balance[label] == pytest.approx(proportion, abs=tolerance)


def test_minority_class_is_present_in_both_subsets(imbalanced_labels: np.ndarray) -> None:
    train, test = stratified_indices(imbalanced_labels)
    assert imbalanced_labels[train].sum() > 0
    assert imbalanced_labels[test].sum() > 0


# ---------------------------------------------------------------------------
# Determinisme
# ---------------------------------------------------------------------------


def test_same_seed_yields_identical_split(imbalanced_labels: np.ndarray) -> None:
    first = stratified_indices(imbalanced_labels, seed=config.SEED)
    second = stratified_indices(imbalanced_labels, seed=config.SEED)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_different_seed_yields_different_split(imbalanced_labels: np.ndarray) -> None:
    """Garde-fou anti faux-positif du test precedent."""
    first, _ = stratified_indices(imbalanced_labels, seed=config.SEED)
    other, _ = stratified_indices(imbalanced_labels, seed=config.SEED + 1)
    assert not np.array_equal(first, other)


def test_split_ignores_global_rng_state(imbalanced_labels: np.ndarray) -> None:
    """`random_state` explicite : un tirage parasite ne doit pas deplacer le split."""
    reference = stratified_indices(imbalanced_labels, seed=config.SEED)

    np.random.seed(999)
    np.random.rand(1_000)  # bruit sur le RNG global
    after_noise = stratified_indices(imbalanced_labels, seed=config.SEED)

    np.testing.assert_array_equal(reference[0], after_noise[0])
    np.testing.assert_array_equal(reference[1], after_noise[1])


# ---------------------------------------------------------------------------
# Empreinte
# ---------------------------------------------------------------------------


def test_fingerprint_is_stable_for_identical_splits(imbalanced_labels: np.ndarray) -> None:
    first = stratified_indices(imbalanced_labels, seed=config.SEED)
    second = stratified_indices(imbalanced_labels, seed=config.SEED)
    assert split_fingerprint(*first) == split_fingerprint(*second)


def test_fingerprint_detects_a_different_split(imbalanced_labels: np.ndarray) -> None:
    first = stratified_indices(imbalanced_labels, seed=config.SEED)
    other = stratified_indices(imbalanced_labels, seed=config.SEED + 1)
    assert split_fingerprint(*first) != split_fingerprint(*other)


def test_fingerprint_is_order_sensitive() -> None:
    """Intervertir train et test doit changer l'empreinte."""
    left, right = np.array([1, 2, 3]), np.array([4, 5])
    assert split_fingerprint(left, right) != split_fingerprint(right, left)


# ---------------------------------------------------------------------------
# Garde-fous
# ---------------------------------------------------------------------------


def test_assert_disjoint_rejects_overlap() -> None:
    with pytest.raises(AssertionError, match="Fuite"):
        assert_disjoint(np.array([0, 1, 2]), np.array([2, 3]))


def test_assert_disjoint_rejects_duplicates() -> None:
    with pytest.raises(AssertionError, match="Doublons"):
        assert_disjoint(np.array([0, 1, 1]), np.array([2, 3]))


def test_empty_labels_are_rejected() -> None:
    with pytest.raises(ValueError, match="vide"):
        stratified_indices(np.array([], dtype=np.int64))


def test_two_dimensional_labels_are_rejected() -> None:
    with pytest.raises(ValueError, match="1-D"):
        stratified_indices(np.zeros((10, 2), dtype=np.int64))


def test_class_balance_sums_to_one(imbalanced_labels: np.ndarray) -> None:
    assert sum(class_balance(imbalanced_labels).values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Persistance
# ---------------------------------------------------------------------------


def test_saved_split_round_trips(imbalanced_labels: np.ndarray, tmp_path) -> None:
    train, test = stratified_indices(imbalanced_labels)
    save_split_indices("demo", train, test, seed=config.SEED, directory=tmp_path)

    loaded_train, loaded_test, seed = load_split_indices("demo", directory=tmp_path)
    np.testing.assert_array_equal(train, loaded_train)
    np.testing.assert_array_equal(test, loaded_test)
    assert seed == config.SEED


def test_loading_a_missing_split_is_explicit(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Aucun split"):
        load_split_indices("absent", directory=tmp_path)


def test_tampered_split_is_detected(tmp_path) -> None:
    """L'empreinte stockee doit invalider un fichier edite a la main."""
    np.savez(
        tmp_path / "corrompu_split.npz",
        train_index=np.array([0, 1]),
        test_index=np.array([2]),
        seed=np.asarray(config.SEED),
        fingerprint=np.asarray("empreinte-mensongere"),
    )
    with pytest.raises(AssertionError, match="incoherente"):
        load_split_indices("corrompu", directory=tmp_path)
