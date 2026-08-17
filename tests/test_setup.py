"""Phase 0 — verifie le socle : config coherente et seeding effectif."""

from __future__ import annotations

import random

import numpy as np
import torch

from src.data import config
from src.data.utils import hash_array, make_generator, set_seed


def test_project_root_points_to_repo() -> None:
    assert (config.PROJECT_ROOT / "requirements.txt").exists()
    assert config.DATA_RAW.parent == config.DATA_DIR


def test_image_specs_are_coherent() -> None:
    assert set(config.IMAGE_DATASETS) == {"fashion_mnist", "cifar10"}
    for name, spec in config.IMAGE_DATASETS.items():
        assert spec.name == name
        assert spec.channels in (1, 3)
        assert spec.native_size <= config.IMAGE_SIZE
        # Un element de normalisation par canal, sinon `Normalize` echoue.
        assert len(spec.norm_mean) == spec.channels
        assert len(spec.norm_std) == spec.channels


def test_normalisation_maps_unit_interval_to_signed_range() -> None:
    """mean=std=0.5 doit envoyer [0, 1] sur exactement [-1, 1]."""
    for spec in config.IMAGE_DATASETS.values():
        mean, std = spec.norm_mean[0], spec.norm_std[0]
        assert (0.0 - mean) / std == config.PIXEL_RANGE[0]
        assert (1.0 - mean) / std == config.PIXEL_RANGE[1]


def test_tabular_config_is_coherent() -> None:
    assert len(config.V_COLUMNS) == 28
    assert config.TARGET_COL not in config.V_COLUMNS
    assert 0.0 < config.TEST_SIZE < 1.0
    assert config.TIME_STRATEGY in ("drop", "cyclical")


def test_fid_config_is_coherent() -> None:
    assert config.FID_FEATURE_DIM in (64, 192, 768, 2048)
    assert config.FID_SPLIT in ("train", "test")
    assert config.FID_BATCH_SIZE > 0


def test_fid_sample_count_supports_a_stable_covariance() -> None:
    """`sigma` est carree de cote FID_FEATURE_DIM : sous ce seuil, elle est singuliere."""
    if config.FID_NUM_SAMPLES is not None:
        assert config.FID_NUM_SAMPLES > config.FID_FEATURE_DIM, (
            f"{config.FID_NUM_SAMPLES} images ne suffisent pas a estimer une "
            f"covariance {config.FID_FEATURE_DIM}x{config.FID_FEATURE_DIM}."
        )


def test_set_seed_makes_all_rng_sources_reproducible() -> None:
    def draw() -> tuple[float, np.ndarray, torch.Tensor]:
        set_seed(config.SEED)
        return random.random(), np.random.rand(5), torch.rand(5)

    py_a, np_a, torch_a = draw()
    py_b, np_b, torch_b = draw()

    assert py_a == py_b
    np.testing.assert_array_equal(np_a, np_b)
    torch.testing.assert_close(torch_a, torch_b)


def test_different_seeds_produce_different_draws() -> None:
    """Garde-fou anti faux-positif : le test precedent doit pouvoir echouer."""
    set_seed(1)
    first = np.random.rand(10)
    set_seed(2)
    second = np.random.rand(10)
    assert not np.array_equal(first, second)


def test_generator_is_independent_from_global_rng() -> None:
    """L'ordre des batchs ne doit pas dependre des tirages faits ailleurs."""
    set_seed(config.SEED)
    reference = torch.randperm(20, generator=make_generator(config.SEED))

    set_seed(config.SEED)
    torch.rand(1000)  # bruit : consomme le RNG global
    after_noise = torch.randperm(20, generator=make_generator(config.SEED))

    torch.testing.assert_close(reference, after_noise)


def test_hash_array_is_stable_and_shape_sensitive() -> None:
    values = np.arange(12, dtype=np.float64)
    assert hash_array(values) == hash_array(values.copy())
    assert hash_array(values) != hash_array(values.reshape(3, 4))
