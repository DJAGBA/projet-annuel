"""Phase 2 — garanties du pipeline tabulaire Credit Card Fraud.

Le test central est `test_scaler_was_fitted_on_train_only` : c'est la garantie
anti-fuite dont depend toute l'interpretation des resultats. Une fuite de
scaler ne provoque aucune erreur visible, seulement des scores flatteurs.

Les tests s'appuient sur un DataFrame synthetique au meme schema (fixture
`synthetic_creditcard`) : ils passent sans credential Kaggle. Ceux qui exigent
le vrai CSV sont marques `requires_download` et skippes s'il est absent.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from src.data import config, tabular
from src.data.splits import class_balance
from src.data.tabular import (
    MANUAL_INSTRUCTIONS,
    build_features,
    describe_imbalance,
    ensure_creditcard_csv,
    extract_minority,
    load_processed,
    load_raw,
    preprocess_tabular,
    save_processed,
)


@pytest.fixture(scope="module")
def split(synthetic_creditcard: pd.DataFrame):
    """Split de reference, calcule une fois pour le module."""
    return preprocess_tabular(frame=synthetic_creditcard)


# ---------------------------------------------------------------------------
# Anti-fuite : le scaler ne voit jamais le test
# ---------------------------------------------------------------------------


def test_scaler_was_fitted_on_train_only(
    split, synthetic_creditcard: pd.DataFrame
) -> None:
    """Les parametres du scaler doivent provenir des seules lignes du train."""
    features, _ = build_features(synthetic_creditcard)
    values = features.to_numpy(dtype=np.float64)

    # 1. Volumetrie : scikit-learn compte les lignes vues pendant le `fit`.
    assert split.scaler.n_samples_seen_ == split.train_index.size
    assert split.scaler.n_samples_seen_ < len(synthetic_creditcard)

    # 2. Valeurs : la moyenne apprise est exactement celle du train recalculee
    #    a part, et differe de celle du jeu complet.
    np.testing.assert_allclose(
        split.scaler.mean_, values[split.train_index].mean(axis=0), rtol=1e-10
    )
    assert not np.allclose(split.scaler.mean_, values.mean(axis=0), rtol=1e-10)


def test_train_is_exactly_centred_but_test_is_not(split) -> None:
    """Consequence observable du fit sur le train seul.

    Si le scaler avait ete ajuste sur l'ensemble des donnees, la moyenne du
    train ne tomberait pas exactement sur zero.
    """
    np.testing.assert_allclose(split.X_train.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(split.X_train.std(axis=0), 1.0, atol=1e-10)
    assert not np.allclose(split.X_test.mean(axis=0), 0.0, atol=1e-6)


def test_test_set_is_only_transformed_never_refitted(
    split, synthetic_creditcard: pd.DataFrame
) -> None:
    """Le test doit etre l'image exacte de `(x - mean_train) / scale_train`."""
    features, _ = build_features(synthetic_creditcard)
    raw_test = features.to_numpy(dtype=np.float64)[split.test_index]
    expected = (raw_test - split.scaler.mean_) / split.scaler.scale_
    np.testing.assert_allclose(split.X_test, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Split stratifie
# ---------------------------------------------------------------------------


def test_split_is_stratified(split, synthetic_creditcard: pd.DataFrame) -> None:
    reference = class_balance(synthetic_creditcard[config.TARGET_COL].to_numpy())
    for subset in (split.y_train, split.y_test):
        balance = class_balance(subset)
        for label, proportion in reference.items():
            assert balance[label] == pytest.approx(proportion, abs=1e-3)


def test_both_subsets_contain_frauds(split) -> None:
    """Un test sans fraude rendrait toute evaluation impossible."""
    assert (split.y_train == config.FRAUD_LABEL).sum() > 0
    assert (split.y_test == config.FRAUD_LABEL).sum() > 0


def test_train_and_test_indices_are_disjoint(split) -> None:
    assert np.intersect1d(split.train_index, split.test_index).size == 0


def test_split_sizes_match_the_configured_ratio(
    split, synthetic_creditcard: pd.DataFrame
) -> None:
    total = len(synthetic_creditcard)
    assert split.X_train.shape[0] + split.X_test.shape[0] == total
    assert split.X_test.shape[0] == pytest.approx(config.TEST_SIZE * total, abs=1)


# ---------------------------------------------------------------------------
# Classe minoritaire isolee depuis le train
# ---------------------------------------------------------------------------


def test_minority_subset_comes_from_train_and_is_pure(split) -> None:
    minority = split.minority_train
    expected = int((split.y_train == config.FRAUD_LABEL).sum())

    assert minority.shape == (expected, split.X_train.shape[1])
    # Chaque ligne extraite doit se retrouver a l'identique dans le train.
    train_rows = {row.tobytes() for row in split.X_train}
    assert all(row.tobytes() in train_rows for row in minority)


def test_minority_subset_excludes_every_test_row(split) -> None:
    test_rows = {row.tobytes() for row in split.X_test}
    assert not any(row.tobytes() in test_rows for row in split.minority_train)


def test_extract_minority_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="alignes"):
        extract_minority(np.zeros((5, 3)), np.zeros(4, dtype=np.int64))


# ---------------------------------------------------------------------------
# Traitement de Time et Amount
# ---------------------------------------------------------------------------


def test_time_is_dropped_by_default(split) -> None:
    assert config.TIME_COL not in split.feature_names
    assert len(split.feature_names) == len(config.V_COLUMNS) + 1  # V1..V28 + Amount


def test_cyclical_strategy_replaces_time_with_sin_cos(
    synthetic_creditcard: pd.DataFrame,
) -> None:
    features, _ = build_features(synthetic_creditcard, time_strategy="cyclical")

    assert config.TIME_COL not in features.columns
    assert {"time_sin", "time_cos"} <= set(features.columns)
    # Encodage sur le cercle unite : pas de discontinuite entre 23h59 et 00h00.
    radius = features["time_sin"] ** 2 + features["time_cos"] ** 2
    np.testing.assert_allclose(radius.to_numpy(), 1.0, atol=1e-12)


def test_unknown_time_strategy_is_rejected(synthetic_creditcard: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="time_strategy"):
        build_features(synthetic_creditcard, time_strategy="keep")  # type: ignore[arg-type]


def test_log1p_reduces_amount_skew(synthetic_creditcard: pd.DataFrame) -> None:
    """Justification chiffree du log1p : c'est la queue lourde qu'il corrige."""
    stats = describe_imbalance(synthetic_creditcard)
    assert stats["amount_skew"] > stats["amount_skew_log1p"]


def test_amount_log1p_can_be_disabled(synthetic_creditcard: pd.DataFrame) -> None:
    features, _ = build_features(synthetic_creditcard, amount_log1p=False)
    np.testing.assert_allclose(
        features[config.AMOUNT_COL].to_numpy(),
        synthetic_creditcard[config.AMOUNT_COL].to_numpy(),
    )


def test_describe_imbalance_reports_the_minority_ratio(
    synthetic_creditcard: pd.DataFrame,
) -> None:
    stats = describe_imbalance(synthetic_creditcard)
    assert stats["n_fraud"] + stats["n_legit"] == stats["n_rows"]
    assert 0.0 < stats["fraud_ratio"] < 0.01
    assert stats["imbalance_ratio"] > 100


# ---------------------------------------------------------------------------
# Ingestion : aucun credential en dur
# ---------------------------------------------------------------------------


def test_missing_csv_raises_actionable_instructions(tmp_path) -> None:
    with pytest.raises(RuntimeError) as failure:
        ensure_creditcard_csv(tmp_path / "absent.csv", allow_download=False)

    message = str(failure.value)
    assert "kaggle.json" in message
    assert "KAGGLE_USERNAME" in message
    assert config.CREDITCARD_KAGGLE_ID in message


def test_instructions_cover_both_credential_sources() -> None:
    assert "KAGGLE_KEY" in MANUAL_INSTRUCTIONS
    assert str(config.CREDITCARD_CSV) in MANUAL_INSTRUCTIONS


def test_source_contains_no_hardcoded_credential() -> None:
    """Aucune valeur de credential ne doit etre assignee dans le code."""
    source = tabular.__file__
    with open(source, encoding="utf-8") as handle:
        code = handle.read()

    forbidden = re.compile(
        r"""(KAGGLE_KEY|KAGGLE_USERNAME|api_key|password|secret)\s*=\s*["'][^"']+["']""",
        re.IGNORECASE,
    )
    assert forbidden.search(code) is None


def test_load_raw_rejects_a_foreign_schema(tmp_path) -> None:
    path = tmp_path / "autre.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Colonnes manquantes"):
        load_raw(path)


# ---------------------------------------------------------------------------
# Persistance
# ---------------------------------------------------------------------------


def test_processed_artifacts_round_trip(split, tmp_path) -> None:
    save_processed(split, directory=tmp_path)
    reloaded = load_processed(directory=tmp_path)

    np.testing.assert_array_equal(reloaded.X_train, split.X_train)
    np.testing.assert_array_equal(reloaded.X_test, split.X_test)
    np.testing.assert_array_equal(reloaded.y_train, split.y_train)
    np.testing.assert_array_equal(reloaded.y_test, split.y_test)
    assert reloaded.feature_names == split.feature_names
    assert reloaded.seed == split.seed
    assert reloaded.fingerprint == split.fingerprint


def test_reloaded_scaler_keeps_its_train_only_parameters(split, tmp_path) -> None:
    """Le scaler persiste porte la garantie anti-fuite ; il ne se refit jamais."""
    save_processed(split, directory=tmp_path)
    reloaded = load_processed(directory=tmp_path)

    np.testing.assert_allclose(reloaded.scaler.mean_, split.scaler.mean_)
    np.testing.assert_allclose(reloaded.scaler.scale_, split.scaler.scale_)
    assert reloaded.scaler.n_samples_seen_ == split.train_index.size


def test_saved_minority_matches_the_train_frauds(split, tmp_path) -> None:
    paths = save_processed(split, directory=tmp_path)
    saved = np.load(paths["X_minority_train"])
    np.testing.assert_array_equal(saved, split.minority_train)


def test_loading_missing_artifacts_is_explicit(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Artefacts tabulaires manquants"):
        load_processed(directory=tmp_path)


def test_summary_is_json_serialisable(split) -> None:
    import json

    payload = json.loads(json.dumps(split.summary()))
    assert payload["n_minority_train"] > 0
    assert len(payload["fingerprint"]) == 64


# ---------------------------------------------------------------------------
# Vrai dataset (skippe si le CSV n'est pas telecharge)
# ---------------------------------------------------------------------------


@pytest.mark.requires_download
def test_real_dataset_has_the_expected_imbalance(creditcard_csv) -> None:
    stats = describe_imbalance(load_raw(creditcard_csv))
    # Valeur de reference publiee pour mlg-ulb/creditcardfraud : ~0.172 %.
    assert stats["fraud_percent"] == pytest.approx(0.172, abs=0.02)


@pytest.mark.requires_download
def test_real_pipeline_produces_a_usable_minority_set(creditcard_csv) -> None:
    result = preprocess_tabular(frame=load_raw(creditcard_csv))
    assert result.minority_train.shape[0] > 300
    assert result.scaler.n_samples_seen_ == result.train_index.size
