"""Phase 4 — statistiques FID de reference.

Le test decisif est `test_extracted_statistics_match_a_direct_computation` :
il confronte l'extraction de `(mu, sigma)` depuis les accumulateurs de
torchmetrics a un calcul direct `mean` / `cov` sur les memes activations. Sans
lui, une erreur d'arithmetique produirait une reference silencieusement fausse,
et tous les FID du projet seraient decales sans que rien ne le signale.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data import config
from src.data.fid_stats import (
    FidStatistics,
    _real_statistics_from_metric,
    compute_real_fid_statistics,
    fid_cache_path,
    get_or_compute_fid_statistics,
    to_uint8_rgb,
)

torchmetrics = pytest.importorskip(
    "torchmetrics.image.fid",
    reason='torchmetrics[image] absent : pip install "torchmetrics[image]>=1.2"',
)
FrechetInceptionDistance = torchmetrics.FrechetInceptionDistance

#: Dimension reduite : meme arithmetique que 2048, caches et matrices minuscules.
TEST_FEATURE_DIM = 64


# ---------------------------------------------------------------------------
# Conversion vers le format Inception
# ---------------------------------------------------------------------------


def test_conversion_maps_signed_range_to_full_byte_range() -> None:
    images = torch.tensor([-1.0, 1.0]).reshape(2, 1, 1, 1)
    converted = to_uint8_rgb(images)

    assert converted.dtype == torch.uint8
    assert int(converted[0].min()) == 0
    assert int(converted[1].max()) == 255


def test_conversion_promotes_grayscale_to_three_identical_channels() -> None:
    """Convention du FID sur images en niveaux de gris : replication, pas coloration."""
    converted = to_uint8_rgb(torch.rand(4, 1, 32, 32) * 2 - 1)

    assert converted.shape == (4, 3, 32, 32)
    torch.testing.assert_close(converted[:, 0], converted[:, 1])
    torch.testing.assert_close(converted[:, 1], converted[:, 2])


def test_conversion_leaves_rgb_channel_count_untouched() -> None:
    assert to_uint8_rgb(torch.rand(4, 3, 32, 32) * 2 - 1).shape == (4, 3, 32, 32)


def test_conversion_clamps_out_of_range_values() -> None:
    converted = to_uint8_rgb(torch.tensor([-1.5, 1.5]).reshape(2, 1, 1, 1))
    assert int(converted.min()) == 0 and int(converted.max()) == 255


@pytest.mark.parametrize("channels", [2, 4])
def test_conversion_rejects_unsupported_channel_counts(channels: int) -> None:
    with pytest.raises(ValueError, match="1 ou 3 canaux"):
        to_uint8_rgb(torch.rand(2, channels, 8, 8))


# ---------------------------------------------------------------------------
# Exactitude de l'extraction (mu, sigma)
# ---------------------------------------------------------------------------


def test_extracted_statistics_match_a_direct_computation() -> None:
    """`(mu, sigma)` doivent egaler `mean` / `cov` sur les memes activations."""
    metric = FrechetInceptionDistance(feature=TEST_FEATURE_DIM, normalize=False)
    images = torch.randint(0, 256, (40, 3, 32, 32), dtype=torch.uint8)
    metric.update(images, real=True)

    mu, sigma = _real_statistics_from_metric(metric)
    with torch.no_grad():
        features = metric.inception(images).double().numpy()

    np.testing.assert_allclose(mu, features.mean(axis=0), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(sigma, np.cov(features, rowvar=False), rtol=1e-5, atol=1e-8)


def test_covariance_is_symmetric_and_positive_semidefinite() -> None:
    metric = FrechetInceptionDistance(feature=TEST_FEATURE_DIM, normalize=False)
    metric.update(torch.randint(0, 256, (40, 3, 32, 32), dtype=torch.uint8), real=True)

    _, sigma = _real_statistics_from_metric(metric)
    np.testing.assert_allclose(sigma, sigma.T, rtol=1e-10)
    # Une covariance valide n'a pas de valeur propre franchement negative.
    assert np.linalg.eigvalsh(sigma).min() > -1e-6


def test_statistics_accumulate_identically_whatever_the_batch_split() -> None:
    """Le decoupage en batchs ne doit pas influencer la reference produite."""
    images = torch.randint(0, 256, (48, 3, 32, 32), dtype=torch.uint8)

    single = FrechetInceptionDistance(feature=TEST_FEATURE_DIM, normalize=False)
    single.update(images, real=True)

    chunked = FrechetInceptionDistance(feature=TEST_FEATURE_DIM, normalize=False)
    for start in range(0, 48, 16):
        chunked.update(images[start : start + 16], real=True)

    mu_single, sigma_single = _real_statistics_from_metric(single)
    mu_chunked, sigma_chunked = _real_statistics_from_metric(chunked)

    np.testing.assert_allclose(mu_single, mu_chunked, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(sigma_single, sigma_chunked, rtol=1e-5, atol=1e-8)


def test_extraction_fails_loudly_if_torchmetrics_changes_its_state() -> None:
    """Le module s'appuie sur des accumulateurs internes : la casse doit etre visible."""

    class Empty:
        pass

    with pytest.raises(AttributeError, match="Accumulateurs torchmetrics"):
        _real_statistics_from_metric(Empty())


# ---------------------------------------------------------------------------
# Clef de cache
# ---------------------------------------------------------------------------


def test_cache_path_distinguishes_full_split_from_subset() -> None:
    full = fid_cache_path("cifar10", n_samples=None)
    subset = fid_cache_path("cifar10", n_samples=5_000)

    assert "full" in full.name
    assert "n5000" in subset.name
    assert full != subset


@pytest.mark.parametrize(
    ("kwargs_a", "kwargs_b"),
    [
        ({"split": "train"}, {"split": "test"}),
        ({"image_size": 32}, {"image_size": 64}),
        ({"feature_dim": 2048}, {"feature_dim": 64}),
        ({"n_samples": 100, "seed": 1}, {"n_samples": 100, "seed": 2}),
    ],
)
def test_every_result_changing_parameter_changes_the_cache_key(
    kwargs_a: dict, kwargs_b: dict
) -> None:
    """Un cache reutilise a tort produirait un FID comparable a rien."""
    assert fid_cache_path("cifar10", **kwargs_a) != fid_cache_path("cifar10", **kwargs_b)


def test_cache_path_separates_datasets() -> None:
    assert fid_cache_path("cifar10") != fid_cache_path("fashion_mnist")


def test_seed_is_absent_from_the_key_when_the_split_is_complete() -> None:
    """Sans sous-echantillonnage, la seed n'influence rien : elle ne doit pas fragmenter le cache."""
    assert fid_cache_path("cifar10", n_samples=None, seed=1) == fid_cache_path(
        "cifar10", n_samples=None, seed=2
    )


# ---------------------------------------------------------------------------
# Persistance
# ---------------------------------------------------------------------------


@pytest.fixture
def statistics() -> FidStatistics:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, TEST_FEATURE_DIM))
    return FidStatistics(
        mu=features.mean(axis=0),
        sigma=np.cov(features, rowvar=False),
        n_samples=200,
        dataset="cifar10",
        split="train",
        image_size=32,
        feature_dim=TEST_FEATURE_DIM,
        seed=None,
    )


def test_statistics_round_trip(statistics: FidStatistics, tmp_path) -> None:
    path = statistics.save(tmp_path / "stats.npz")
    reloaded = FidStatistics.load(path)

    np.testing.assert_allclose(reloaded.mu, statistics.mu)
    np.testing.assert_allclose(reloaded.sigma, statistics.sigma)
    assert reloaded.metadata() == statistics.metadata()


def test_loading_a_missing_cache_is_explicit(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Aucune statistique FID"):
        FidStatistics.load(tmp_path / "absent.npz")


def test_save_creates_missing_parent_directories(
    statistics: FidStatistics, tmp_path
) -> None:
    path = statistics.save(tmp_path / "a" / "b" / "stats.npz")
    assert path.exists()


# ---------------------------------------------------------------------------
# Calcul sur vraies images (skippe si le dataset n'est pas telecharge)
# ---------------------------------------------------------------------------


@pytest.mark.requires_download
@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(config.IMAGE_DATASETS))
def test_real_statistics_have_the_expected_shapes(name: str, image_loaders) -> None:
    image_loaders(name)  # skippe proprement si le dataset est absent

    statistics = compute_real_fid_statistics(
        name,
        split="test",
        n_samples=32,
        feature_dim=TEST_FEATURE_DIM,
        batch_size=16,
        num_workers=0,
        download=False,
        progress=False,
    )

    assert statistics.mu.shape == (TEST_FEATURE_DIM,)
    assert statistics.sigma.shape == (TEST_FEATURE_DIM, TEST_FEATURE_DIM)
    assert statistics.n_samples == 32
    assert statistics.dataset == name


@pytest.mark.requires_download
@pytest.mark.slow
def test_cache_is_written_then_reused(image_loaders, tmp_path) -> None:
    image_loaders("fashion_mnist")

    common = dict(
        split="test",
        n_samples=32,
        feature_dim=TEST_FEATURE_DIM,
        directory=tmp_path,
        batch_size=16,
        num_workers=0,
        download=False,
        progress=False,
    )
    first = get_or_compute_fid_statistics("fashion_mnist", **common)
    path = fid_cache_path(
        "fashion_mnist",
        split="test",
        n_samples=32,
        feature_dim=TEST_FEATURE_DIM,
        directory=tmp_path,
    )
    assert path.exists()

    second = get_or_compute_fid_statistics("fashion_mnist", **common)
    np.testing.assert_allclose(first.mu, second.mu)
    np.testing.assert_allclose(first.sigma, second.sigma)


@pytest.mark.requires_download
@pytest.mark.slow
def test_subsampling_is_reproducible_at_fixed_seed(image_loaders, tmp_path) -> None:
    """Deux calculs a meme seed doivent voir exactement les memes images."""
    image_loaders("fashion_mnist")

    common = dict(
        split="test",
        n_samples=32,
        feature_dim=TEST_FEATURE_DIM,
        batch_size=16,
        num_workers=0,
        download=False,
        progress=False,
    )
    first = compute_real_fid_statistics("fashion_mnist", seed=config.SEED, **common)
    second = compute_real_fid_statistics("fashion_mnist", seed=config.SEED, **common)
    other = compute_real_fid_statistics("fashion_mnist", seed=config.SEED + 1, **common)

    np.testing.assert_allclose(first.mu, second.mu)
    assert not np.allclose(first.mu, other.mu)
