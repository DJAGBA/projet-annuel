"""Phase 1 — garanties du pipeline images.

Deux familles de tests :

* ceux qui s'appliquent aux transforms sur des images PIL synthetiques : ils
  couvrent la plage, les shapes et les canaux **sans aucun telechargement** ;
* ceux qui parcourent les vrais datasets torchvision, skippes s'ils ne sont pas
  sur disque.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from src.data import config
from src.data.images import (
    build_transform,
    denormalize,
    get_image_dataloader,
    resolve_spec,
    sanity_check_batch,
    save_sample_grid,
)

DATASET_NAMES = sorted(config.IMAGE_DATASETS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gradient_image(size: int, channels: int) -> Image.Image:
    """Image PIL couvrant exactement [0, 255], pour tester les bornes."""
    count = size * size * channels
    flat = np.linspace(0, 255, count).round().astype(np.uint8)
    if channels == 1:
        return Image.fromarray(flat.reshape(size, size), mode="L")
    return Image.fromarray(flat.reshape(size, size, channels), mode="RGB")


def _uniform_image(size: int, channels: int, value: int) -> Image.Image:
    """Image PIL constante, pour isoler l'effet du padding."""
    shape = (size, size) if channels == 1 else (size, size, channels)
    array = np.full(shape, value, dtype=np.uint8)
    return Image.fromarray(array, mode="L" if channels == 1 else "RGB")


def _valid_batch(batch: int = 4, channels: int = 1, size: int = 32) -> torch.Tensor:
    """Batch synthetique conforme au contrat du pipeline."""
    return torch.rand(batch, channels, size, size) * 2.0 - 1.0


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------


def test_resolve_spec_lists_available_names_on_error() -> None:
    with pytest.raises(KeyError, match="fashion_mnist"):
        resolve_spec("mnist")


@pytest.mark.parametrize("name", DATASET_NAMES)
def test_resolve_spec_returns_matching_spec(name: str) -> None:
    assert resolve_spec(name).name == name


# ---------------------------------------------------------------------------
# Transforms : plage [-1, 1]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", DATASET_NAMES)
def test_transform_maps_pixel_extremes_to_signed_unit_range(name: str) -> None:
    """0 -> -1 et 255 -> +1 exactement : la `tanh` du generateur couvre ce support."""
    spec = resolve_spec(name)
    transform = build_transform(spec, config.IMAGE_SIZE)
    tensor = transform(_gradient_image(spec.native_size, spec.channels))

    low, high = config.PIXEL_RANGE
    assert torch.isclose(tensor.min(), torch.tensor(low))
    assert torch.isclose(tensor.max(), torch.tensor(high))


@pytest.mark.parametrize("name", DATASET_NAMES)
def test_transform_output_is_float32(name: str) -> None:
    spec = resolve_spec(name)
    tensor = build_transform(spec, config.IMAGE_SIZE)(
        _gradient_image(spec.native_size, spec.channels)
    )
    assert tensor.dtype == torch.float32


def test_midpoint_pixel_maps_to_zero() -> None:
    """Un gris moyen doit tomber au centre de [-1, 1], pas ailleurs."""
    spec = resolve_spec("cifar10")
    tensor = build_transform(spec, config.IMAGE_SIZE)(
        _uniform_image(spec.native_size, spec.channels, 128)
    )
    assert abs(float(tensor.mean())) < 0.01


# ---------------------------------------------------------------------------
# Transforms : geometrie et canaux
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", DATASET_NAMES)
def test_transform_produces_target_shape(name: str) -> None:
    spec = resolve_spec(name)
    tensor = build_transform(spec, config.IMAGE_SIZE)(
        _gradient_image(spec.native_size, spec.channels)
    )
    assert tensor.shape == (spec.channels, config.IMAGE_SIZE, config.IMAGE_SIZE)


def test_fashion_mnist_is_padded_from_28_to_32() -> None:
    spec = resolve_spec("fashion_mnist")
    assert spec.native_size == 28
    tensor = build_transform(spec, 32)(_gradient_image(28, 1))
    assert tensor.shape == (1, 32, 32)


def test_padding_adds_a_black_border_not_a_stretched_image() -> None:
    """La bordure ajoutee vaut -1 (noir) et le contenu reste intact."""
    spec = resolve_spec("fashion_mnist")
    tensor = build_transform(spec, 32)(_uniform_image(28, 1, 255))

    border, interior = tensor[0, 0, :], tensor[0, 2:30, 2:30]
    assert torch.allclose(border, torch.full_like(border, -1.0))
    assert torch.allclose(interior, torch.full_like(interior, 1.0))


def test_downscaling_falls_back_to_resize() -> None:
    spec = resolve_spec("cifar10")
    tensor = build_transform(spec, 16)(_gradient_image(32, 3))
    assert tensor.shape == (3, 16, 16)


@pytest.mark.parametrize(("name", "forced"), [("fashion_mnist", 3), ("cifar10", 1)])
def test_channel_override_changes_channel_count(name: str, forced: int) -> None:
    """Un seul chemin de code sert 1 et 3 canaux, dans les deux sens."""
    spec = resolve_spec(name)
    tensor = build_transform(spec, config.IMAGE_SIZE, channels=forced)(
        _gradient_image(spec.native_size, spec.channels)
    )
    assert tensor.shape[0] == forced
    low, high = config.PIXEL_RANGE
    assert tensor.min() >= low - 1e-5 and tensor.max() <= high + 1e-5


def test_grayscale_promoted_to_rgb_replicates_the_channel() -> None:
    spec = resolve_spec("fashion_mnist")
    tensor = build_transform(spec, 32, channels=3)(_gradient_image(28, 1))
    torch.testing.assert_close(tensor[0], tensor[1])
    torch.testing.assert_close(tensor[1], tensor[2])


@pytest.mark.parametrize("channels", [0, 2, 4, -1])
def test_invalid_channel_count_is_rejected(channels: int) -> None:
    with pytest.raises(ValueError, match="channels"):
        build_transform(resolve_spec("cifar10"), config.IMAGE_SIZE, channels=channels)


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def test_sanity_check_accepts_a_conforming_batch() -> None:
    stats = sanity_check_batch(
        _valid_batch(), expected_channels=1, expected_size=32
    )
    assert set(stats) == {"min", "max", "mean", "std"}


def test_sanity_check_rejects_zero_one_normalisation() -> None:
    """Le piege que le pipeline doit attraper : une normalisation [0, 1]."""
    with pytest.raises(AssertionError, match="amplitude|borne"):
        sanity_check_batch(
            torch.rand(4, 1, 32, 32), expected_channels=1, expected_size=32
        )


def test_sanity_check_rejects_out_of_range_values() -> None:
    batch = _valid_batch()
    batch[0, 0, 0, 0] = 1.5
    with pytest.raises(AssertionError, match="borne"):
        sanity_check_batch(batch, expected_channels=1, expected_size=32)


def test_sanity_check_rejects_wrong_channel_count() -> None:
    with pytest.raises(AssertionError, match="canal|canaux"):
        sanity_check_batch(
            _valid_batch(channels=3), expected_channels=1, expected_size=32
        )


def test_sanity_check_rejects_wrong_spatial_size() -> None:
    with pytest.raises(AssertionError, match="Taille"):
        sanity_check_batch(
            _valid_batch(size=28), expected_channels=1, expected_size=32
        )


def test_sanity_check_rejects_degenerate_batch() -> None:
    with pytest.raises(AssertionError, match="degenere"):
        sanity_check_batch(
            torch.zeros(4, 1, 32, 32), expected_channels=1, expected_size=32
        )


def test_sanity_check_rejects_unbatched_tensor() -> None:
    with pytest.raises(AssertionError, match=r"\(B, C, H, W\)"):
        sanity_check_batch(
            torch.zeros(1, 32, 32), expected_channels=1, expected_size=32
        )


# ---------------------------------------------------------------------------
# Denormalisation & grille
# ---------------------------------------------------------------------------


def test_denormalize_maps_signed_range_to_unit_range() -> None:
    source = torch.tensor([-1.0, 0.0, 1.0])
    torch.testing.assert_close(denormalize(source), torch.tensor([0.0, 0.5, 1.0]))


def test_denormalize_clamps_generator_overshoot() -> None:
    """Une sortie de generateur peut deborder : la grille ne doit pas saturer faux."""
    clamped = denormalize(torch.tensor([-1.4, 1.4]))
    assert clamped.min() >= 0.0 and clamped.max() <= 1.0


def test_save_sample_grid_writes_a_png(tmp_path) -> None:
    destination = save_sample_grid(_valid_batch(batch=8), tmp_path / "sub" / "grid.png")
    assert destination.exists() and destination.stat().st_size > 0


# ---------------------------------------------------------------------------
# Vrais datasets (skippes si non telecharges)
# ---------------------------------------------------------------------------


@pytest.mark.requires_download
@pytest.mark.parametrize("name", DATASET_NAMES)
def test_real_batch_respects_the_pipeline_contract(name: str, image_loaders) -> None:
    spec = resolve_spec(name)
    train_loader, test_loader = image_loaders(name)

    for loader in (train_loader, test_loader):
        images, labels = next(iter(loader))
        sanity_check_batch(
            images,
            expected_channels=spec.channels,
            expected_size=config.IMAGE_SIZE,
        )
        assert labels.shape[0] == images.shape[0]
        assert int(labels.max()) < spec.num_classes


@pytest.mark.requires_download
@pytest.mark.parametrize("name", DATASET_NAMES)
def test_train_and_test_splits_are_distinct_sizes(name: str, image_loaders) -> None:
    train_loader, test_loader = image_loaders(name)
    assert len(train_loader.dataset) > len(test_loader.dataset) > 0


@pytest.mark.requires_download
def test_batch_order_is_reproducible_at_fixed_seed(image_loaders) -> None:
    """Condition necessaire pour comparer DCGAN et WGAN-GP sur la meme sequence."""
    first, _ = image_loaders("fashion_mnist", seed=config.SEED)
    second, _ = image_loaders("fashion_mnist", seed=config.SEED)
    torch.testing.assert_close(next(iter(first))[0], next(iter(second))[0])


@pytest.mark.requires_download
def test_batch_order_changes_with_the_seed(image_loaders) -> None:
    """Garde-fou anti faux-positif du test precedent."""
    first, _ = image_loaders("fashion_mnist", seed=config.SEED)
    other, _ = image_loaders("fashion_mnist", seed=config.SEED + 1)
    assert not torch.equal(next(iter(first))[0], next(iter(other))[0])


@pytest.mark.requires_download
def test_test_loader_is_not_shuffled(image_loaders) -> None:
    """Le test doit etre parcouru identiquement a chaque evaluation."""
    _, first = image_loaders("fashion_mnist", seed=config.SEED)
    _, second = image_loaders("fashion_mnist", seed=config.SEED + 1)
    torch.testing.assert_close(next(iter(first))[0], next(iter(second))[0])
