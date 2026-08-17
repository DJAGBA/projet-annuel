"""Phase 3 — garde-fous anti-fuite autour de l'augmentation synthetique.

Invariant defendu ici : **le jeu de test reste 100 % reel**. Un test contamine
par des echantillons issus du generateur ferait noter le modele par lui-meme,
sans qu'aucune erreur ne se declenche.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import config
from src.data.augment import (
    assert_no_synthetic_in_test,
    assert_rows_are_disjoint,
    assert_test_unchanged,
    inject_synthetic,
    load_synthetic,
    row_signatures,
    save_synthetic,
)
from src.data.tabular import preprocess_tabular


@pytest.fixture(scope="module")
def split(synthetic_creditcard: pd.DataFrame):
    return preprocess_tabular(frame=synthetic_creditcard)


@pytest.fixture
def fake_generated(split) -> np.ndarray:
    """Faux lot "genere" : bruit dans l'espace des features scalees."""
    rng = np.random.default_rng(123)
    return rng.normal(0.0, 1.0, size=(50, split.X_train.shape[1]))


# ---------------------------------------------------------------------------
# L'invariant central
# ---------------------------------------------------------------------------


def test_test_set_contains_no_synthetic_sample(split, fake_generated) -> None:
    inject_synthetic(split.X_train, split.y_train, fake_generated)
    assert_no_synthetic_in_test(split.X_test, fake_generated)


def test_augmentation_never_touches_the_test_set(split, fake_generated) -> None:
    """Le test doit etre bit a bit identique avant et apres augmentation."""
    before = split.X_test.copy()
    inject_synthetic(split.X_train, split.y_train, fake_generated)
    assert_test_unchanged(before, split.X_test)


def test_every_augmented_row_is_either_train_or_synthetic(
    split, fake_generated
) -> None:
    """Aucune ligne du train augmente ne peut provenir du test."""
    augmented = inject_synthetic(split.X_train, split.y_train, fake_generated)

    test_rows = row_signatures(split.X_test)
    train_rows = row_signatures(split.X_train)
    synthetic_rows = row_signatures(fake_generated)

    for row in np.ascontiguousarray(augmented.X):
        signature = row.tobytes()
        assert signature not in test_rows
        assert signature in train_rows or signature in synthetic_rows


def test_guard_detects_a_deliberately_contaminated_test(split, fake_generated) -> None:
    """Garde-fou anti faux-positif : le controle doit savoir echouer."""
    contaminated = np.concatenate([split.X_test, fake_generated[:3]], axis=0)
    with pytest.raises(AssertionError, match="Fuite"):
        assert_no_synthetic_in_test(contaminated, fake_generated)


def test_guard_detects_a_modified_test_set(split) -> None:
    modified = split.X_test.copy()
    modified[0, 0] += 1e-6
    with pytest.raises(AssertionError, match="contenu du test"):
        assert_test_unchanged(split.X_test, modified)


def test_guard_detects_a_resized_test_set(split) -> None:
    with pytest.raises(AssertionError, match="change de forme"):
        assert_test_unchanged(split.X_test, split.X_test[:-1])


# ---------------------------------------------------------------------------
# Comptes et tracabilite de l'injection
# ---------------------------------------------------------------------------


def test_injection_adds_exactly_the_expected_rows(split, fake_generated) -> None:
    augmented = inject_synthetic(split.X_train, split.y_train, fake_generated)

    assert augmented.X.shape[0] == split.X_train.shape[0] + fake_generated.shape[0]
    assert augmented.n_synthetic == fake_generated.shape[0]
    assert augmented.n_real == split.X_train.shape[0]


def test_synthetic_rows_are_labelled_as_fraud(split, fake_generated) -> None:
    augmented = inject_synthetic(split.X_train, split.y_train, fake_generated)
    _, y_synthetic = augmented.synthetic_subset()
    assert np.all(y_synthetic == config.FRAUD_LABEL)


def test_real_subset_is_recoverable_after_shuffling(split, fake_generated) -> None:
    """Le masque `is_synthetic` doit survivre au melange."""
    augmented = inject_synthetic(split.X_train, split.y_train, fake_generated)
    X_real, y_real = augmented.real_subset()

    assert X_real.shape == split.X_train.shape
    # Meme contenu, ordre different : on compare les ensembles de lignes.
    assert row_signatures(X_real) == row_signatures(split.X_train)
    assert sorted(y_real.tolist()) == sorted(split.y_train.tolist())


def test_shuffle_disperses_synthetic_rows(split, fake_generated) -> None:
    """Sans melange, les derniers batchs d'une epoque seraient tous synthetiques."""
    augmented = inject_synthetic(split.X_train, split.y_train, fake_generated)
    positions = np.flatnonzero(augmented.is_synthetic)
    assert positions.min() < split.X_train.shape[0] / 2


def test_unshuffled_injection_keeps_synthetic_rows_at_the_end(
    split, fake_generated
) -> None:
    augmented = inject_synthetic(
        split.X_train, split.y_train, fake_generated, shuffle=False
    )
    assert np.all(augmented.is_synthetic[-fake_generated.shape[0] :])
    assert not np.any(augmented.is_synthetic[: -fake_generated.shape[0]])


def test_empty_synthetic_set_is_the_control_run(split) -> None:
    """0 % d'augmentation : le train doit contenir les seules lignes reelles.

    Le run de controle est melange comme les runs augmentes : c'est ce qui rend
    la comparaison honnete, l'ordre des lignes n'etant alors plus une variable
    qui differe entre les deux protocoles.
    """
    augmented = inject_synthetic(
        split.X_train, split.y_train, np.empty((0, split.X_train.shape[1]))
    )
    assert augmented.n_synthetic == 0
    assert augmented.synthetic_ratio == 0.0
    assert augmented.X.shape == split.X_train.shape
    assert row_signatures(augmented.X) == row_signatures(split.X_train)


def test_control_run_is_unshuffled_on_demand(split) -> None:
    """`shuffle=False` rend le run de controle strictement identique au train."""
    augmented = inject_synthetic(
        split.X_train,
        split.y_train,
        np.empty((0, split.X_train.shape[1])),
        shuffle=False,
    )
    np.testing.assert_array_equal(augmented.X, split.X_train)
    np.testing.assert_array_equal(augmented.y, split.y_train)


def test_summary_reports_the_augmentation_ratio(split, fake_generated) -> None:
    augmented = inject_synthetic(split.X_train, split.y_train, fake_generated)
    summary = augmented.summary()
    assert summary["n_total"] == augmented.X.shape[0]
    assert summary["synthetic_ratio"] == pytest.approx(
        fake_generated.shape[0] / augmented.X.shape[0]
    )


# ---------------------------------------------------------------------------
# Validation des entrees venues du perimetre ML
# ---------------------------------------------------------------------------


def test_diverged_generator_output_is_rejected(split) -> None:
    """Un WGAN-GP qui diverge emet des NaN : ils ne doivent pas entrer dans le train."""
    poisoned = np.zeros((5, split.X_train.shape[1]))
    poisoned[2, 0] = np.nan
    poisoned[3, 1] = np.inf

    with pytest.raises(ValueError, match="non finies"):
        inject_synthetic(split.X_train, split.y_train, poisoned)


def test_feature_mismatch_is_rejected(split) -> None:
    with pytest.raises(ValueError, match="features"):
        inject_synthetic(
            split.X_train, split.y_train, np.zeros((5, split.X_train.shape[1] + 1))
        )


def test_misaligned_labels_are_rejected(split) -> None:
    with pytest.raises(ValueError, match="alignes"):
        inject_synthetic(split.X_train, split.y_train[:-1], np.zeros((0, 1)))


def test_assert_rows_are_disjoint_tolerates_empty_matrices() -> None:
    assert_rows_are_disjoint(np.zeros((0, 3)), np.ones((4, 3)))


# ---------------------------------------------------------------------------
# Echange de fichiers avec le perimetre ML
# ---------------------------------------------------------------------------


def test_synthetic_batch_round_trips(fake_generated, tmp_path) -> None:
    save_synthetic(fake_generated, "demo", directory=tmp_path)
    loaded = load_synthetic(
        "demo", n_features=fake_generated.shape[1], directory=tmp_path
    )
    np.testing.assert_array_equal(loaded, fake_generated)


def test_loading_a_missing_batch_is_explicit(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Aucun lot synthetique"):
        load_synthetic("absent", directory=tmp_path)


def test_loading_validates_the_feature_count(fake_generated, tmp_path) -> None:
    save_synthetic(fake_generated, "demo", directory=tmp_path)
    with pytest.raises(ValueError, match="features"):
        load_synthetic("demo", n_features=999, directory=tmp_path)
