"""Pipeline images pour l'entrainement des GANs (Fashion-MNIST / CIFAR-10).

Un unique chemin de code sert les deux datasets : tout ce qui les distingue
(nombre de canaux, taille native, classe torchvision) vit dans
`config.ImageDatasetSpec`. Ajouter un dataset revient a ajouter une entree dans
`config.IMAGE_DATASETS`, pas une branche `if` dans ce module.

Point critique : les tenseurs sortent normalises dans **[-1, 1]**, pas [0, 1].
Le generateur se termine par une `tanh` ; avec une normalisation [0, 1] le
discriminateur comparerait des vraies et des fausses images vivant sur des
supports differents, et le GAN convergerait vers cet artefact plutot que vers
la distribution des donnees.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image

from src.data import config
from src.data.config import ImageDatasetSpec
from src.data.utils import (
    ensure_dirs,
    hash_array,
    make_generator,
    resolve_pin_memory,
    seed_worker,
)

__all__ = [
    "resolve_spec",
    "build_transform",
    "get_image_datasets",
    "get_image_dataloader",
    "denormalize",
    "sanity_check_batch",
    "batch_order_fingerprint",
    "save_sample_grid",
]

#: Valeur de remplissage du padding. Le fond de Fashion-MNIST est noir (0), donc
#: un padding a 0 est invisible : la bordure ajoutee prolonge le fond au lieu
#: d'introduire un cadre que le discriminateur apprendrait a detecter.
PAD_FILL: int = 0


# ---------------------------------------------------------------------------
# Specs & transforms
# ---------------------------------------------------------------------------


def resolve_spec(dataset_name: str) -> ImageDatasetSpec:
    """Retourne la spec d'un dataset image.

    Args:
        dataset_name: Cle de `config.IMAGE_DATASETS` (ex. `"fashion_mnist"`).

    Returns:
        La `ImageDatasetSpec` correspondante.

    Raises:
        KeyError: Si le nom est inconnu. Le message liste les noms valides.
    """
    try:
        return config.IMAGE_DATASETS[dataset_name]
    except KeyError as exc:
        available = ", ".join(sorted(config.IMAGE_DATASETS))
        raise KeyError(
            f"Dataset image inconnu : {dataset_name!r}. Disponibles : {available}."
        ) from exc


def _resolve_channels(spec: ImageDatasetSpec, channels: int | None) -> int:
    """Determine le nombre de canaux effectif (surcharge optionnelle de la spec)."""
    effective = spec.channels if channels is None else channels
    if effective not in (1, 3):
        raise ValueError(f"channels doit valoir 1 ou 3, recu {effective!r}.")
    return effective


def _geometry_transform(native_size: int, image_size: int) -> list[torch.nn.Module]:
    """Construit l'etape de mise a la taille cible.

    Le padding est prefere au redimensionnement quand la cible est plus grande :
    il preserve exactement les statistiques de pixels, la ou une interpolation
    introduirait un flou que le GAN apprendrait comme une caracteristique des
    vraies images.

    Args:
        native_size: Cote de l'image brute.
        image_size: Cote cible.

    Returns:
        Une liste de 0 ou 1 transform (vide si les tailles coincident).
    """
    if image_size == native_size:
        return []

    if image_size > native_size:
        delta = image_size - native_size
        before = delta // 2
        after = delta - before
        # torchvision.Pad attend (left, top, right, bottom).
        return [transforms.Pad((before, before, after, after), fill=PAD_FILL)]

    return [transforms.Resize((image_size, image_size), antialias=True)]


def build_transform(
    spec: ImageDatasetSpec,
    image_size: int = config.IMAGE_SIZE,
    *,
    channels: int | None = None,
) -> transforms.Compose:
    """Assemble la chaine de transforms d'un dataset image.

    Etapes : harmonisation des canaux -> mise a la taille cible -> `ToTensor`
    ([0, 255] uint8 -> [0, 1] float32) -> `Normalize` ([0, 1] -> [-1, 1]).

    Args:
        spec: Spec du dataset source.
        image_size: Cote cible en pixels.
        channels: Force un nombre de canaux different de la spec (1 ou 3).
            Utile pour entrainer une meme architecture sur les deux datasets :
            `channels=3` replique le canal de Fashion-MNIST. `None` conserve
            les canaux natifs.

    Returns:
        La `Compose` a passer au dataset torchvision.
    """
    effective_channels = _resolve_channels(spec, channels)

    steps: list[torch.nn.Module] = []
    if effective_channels != spec.channels:
        # 1 -> 3 replique le canal ; 3 -> 1 applique la luminance.
        steps.append(transforms.Grayscale(num_output_channels=effective_channels))

    steps.extend(_geometry_transform(spec.native_size, image_size))
    steps.append(transforms.ToTensor())

    # On repart du scalaire de la spec pour ne pas redefinir 0.5 ici : la
    # longueur doit suivre les canaux *effectifs*, pas ceux du dataset source.
    mean = (spec.norm_mean[0],) * effective_channels
    std = (spec.norm_std[0],) * effective_channels
    steps.append(transforms.Normalize(mean, std))

    return transforms.Compose(steps)


# ---------------------------------------------------------------------------
# Datasets & DataLoaders
# ---------------------------------------------------------------------------


def get_image_datasets(
    dataset_name: str,
    image_size: int = config.IMAGE_SIZE,
    *,
    channels: int | None = None,
    root: Path = config.DATA_RAW,
    download: bool = True,
) -> tuple[Dataset, Dataset]:
    """Instancie les datasets train et test torchvision, transforms appliques.

    Args:
        dataset_name: Cle de `config.IMAGE_DATASETS`.
        image_size: Cote cible en pixels.
        channels: Surcharge du nombre de canaux (cf. `build_transform`).
        root: Repertoire de telechargement / cache.
        download: Telecharge si absent du disque.

    Returns:
        Le couple `(train_dataset, test_dataset)`.
    """
    spec = resolve_spec(dataset_name)
    dataset_cls = getattr(torchvision.datasets, spec.torchvision_class)
    transform = build_transform(spec, image_size, channels=channels)

    root.mkdir(parents=True, exist_ok=True)
    train = dataset_cls(
        root=str(root), train=True, download=download, transform=transform
    )
    test = dataset_cls(
        root=str(root), train=False, download=download, transform=transform
    )
    return train, test


def get_image_dataloader(
    dataset_name: str,
    image_size: int = config.IMAGE_SIZE,
    batch_size: int = config.BATCH_SIZE,
    *,
    channels: int | None = None,
    num_workers: int = config.NUM_WORKERS,
    seed: int = config.SEED,
    root: Path = config.DATA_RAW,
    download: bool = True,
    drop_last: bool | None = None,
    shuffle_train: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Factory publique : DataLoaders train / test prets pour l'entrainement GAN.

    Le shuffle du train est pilote par un `torch.Generator` dedie et les workers
    sont re-seedes : a seed fixee, l'ordre des batchs est reproductible d'un run
    a l'autre, condition necessaire pour comparer DCGAN et WGAN-GP sur des
    sequences de donnees identiques.

    Args:
        dataset_name: Cle de `config.IMAGE_DATASETS`.
        image_size: Cote cible en pixels.
        batch_size: Taille de batch.
        channels: Surcharge du nombre de canaux (cf. `build_transform`).
        num_workers: Process de chargement. 0 sous Windows par defaut.
        seed: Graine du generateur de shuffle.
        root: Repertoire de telechargement / cache.
        download: Telecharge si absent du disque.
        drop_last: Jette le dernier batch incomplet du train. `None` reprend
            `config.DROP_LAST` : des batchs de taille constante evitent des
            statistiques de BatchNorm bruitees en fin d'epoque.
        shuffle_train: Melange le train. A desactiver pour un debug ordonne.

    Returns:
        Le couple `(train_loader, test_loader)`. Le test n'est ni melange ni
        tronque : il doit rester integralement et identiquement parcouru.
    """
    train_ds, test_ds = get_image_datasets(
        dataset_name,
        image_size,
        channels=channels,
        root=root,
        download=download,
    )

    if drop_last is None:
        drop_last = config.DROP_LAST

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": resolve_pin_memory(),
        "worker_init_fn": seed_worker,
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(
        train_ds,
        shuffle=shuffle_train,
        drop_last=drop_last,
        generator=make_generator(seed),
        **common,
    )
    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Sanity checks & inspection visuelle
# ---------------------------------------------------------------------------


def denormalize(images: torch.Tensor) -> torch.Tensor:
    """Ramene un tenseur de [-1, 1] vers [0, 1] pour affichage ou sauvegarde.

    Args:
        images: Tenseur normalise, de forme `(B, C, H, W)` ou `(C, H, W)`.

    Returns:
        Le tenseur dans [0, 1], borne par `clamp` (les sorties de generateur
        peuvent deborder legerement).
    """
    return ((images + 1.0) / 2.0).clamp(0.0, 1.0)


def sanity_check_batch(
    images: torch.Tensor,
    *,
    expected_channels: int,
    expected_size: int,
    tol: float = 1e-5,
) -> dict[str, float]:
    """Verifie qu'un batch respecte le contrat du pipeline.

    Controle la forme, le nombre de canaux, le dtype, la plage [-1, 1] et le
    fait que le batch ne soit pas degenere (image constante).

    Args:
        images: Batch de forme `(B, C, H, W)`.
        expected_channels: Nombre de canaux attendu.
        expected_size: Cote attendu.
        tol: Tolerance sur les bornes, pour absorber l'erreur flottante.

    Returns:
        Un dict de statistiques (`min`, `max`, `mean`, `std`) utile en log.

    Raises:
        AssertionError: Si l'une des garanties est violee.
    """
    assert images.ndim == 4, f"Batch attendu en (B, C, H, W), recu {tuple(images.shape)}"
    assert images.dtype == torch.float32, f"dtype attendu float32, recu {images.dtype}"

    _, channels, height, width = images.shape
    assert channels == expected_channels, (
        f"{expected_channels} canal/canaux attendu(s), recu {channels}"
    )
    assert (height, width) == (expected_size, expected_size), (
        f"Taille attendue {expected_size}x{expected_size}, recue {height}x{width}"
    )

    low, high = config.PIXEL_RANGE
    minimum = float(images.min())
    maximum = float(images.max())
    assert minimum >= low - tol, f"Valeur {minimum} sous la borne {low}"
    assert maximum <= high + tol, f"Valeur {maximum} au-dessus de la borne {high}"

    # Un pipeline casse (mauvaise normalisation, images vides) produit un batch
    # quasi constant : on exige qu'il couvre au moins la moitie de la plage.
    span = maximum - minimum
    assert span > (high - low) / 2, f"Batch degenere : amplitude {span:.4f}"

    return {
        "min": minimum,
        "max": maximum,
        "mean": float(images.mean()),
        "std": float(images.std()),
    }


def batch_order_fingerprint(loader: DataLoader, n_batches: int = 3) -> str:
    """Empreinte des premiers batchs servis par un `DataLoader`.

    Equivalent, pour le pipeline images, de `splits.split_fingerprint` : permet
    d'affirmer que deux runs (DCGAN et WGAN-GP, par exemple) ont consomme la
    meme sequence de donnees, sans avoir a conserver les images elles-memes.

    Note:
        Iterer un `DataLoader` melange fait avancer son `torch.Generator`. Cette
        fonction doit donc etre appelee sur un loader **neuf** pour etre
        comparable a un autre appel -- typiquement juste apres construction.

    Args:
        loader: DataLoader a inspecter.
        n_batches: Nombre de batchs a integrer a l'empreinte.

    Returns:
        L'empreinte hexadecimale de la sequence.
    """
    digest = hashlib.sha256()
    for index, (images, labels) in enumerate(loader):
        if index >= n_batches:
            break
        digest.update(hash_array(images.numpy()).encode())
        digest.update(hash_array(labels.numpy()).encode())
    return digest.hexdigest()


def save_sample_grid(
    images: torch.Tensor,
    path: Path,
    *,
    nrow: int = 8,
) -> Path:
    """Sauvegarde une grille d'echantillons pour verification visuelle.

    Args:
        images: Batch normalise dans [-1, 1], de forme `(B, C, H, W)`.
        path: Fichier de sortie (PNG). Les parents sont crees au besoin.
        nrow: Nombre d'images par ligne.

    Returns:
        Le chemin ecrit.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(denormalize(images), path, nrow=nrow)
    return path


def prepare_dataset(
    dataset_name: str,
    image_size: int = config.IMAGE_SIZE,
    *,
    batch_size: int = config.BATCH_SIZE,
    channels: int | None = None,
    save_grid: bool = True,
) -> dict[str, float]:
    """Telecharge, valide et illustre un dataset image en une passe.

    Point d'entree pratique pour l'etape "preparation des donnees" : construit
    les loaders, tire un batch, applique `sanity_check_batch` et ecrit une
    grille d'echantillons dans `config.SAMPLES_DIR`.

    Args:
        dataset_name: Cle de `config.IMAGE_DATASETS`.
        image_size: Cote cible en pixels.
        batch_size: Taille de batch.
        channels: Surcharge du nombre de canaux.
        save_grid: Ecrit la grille de verification visuelle.

    Returns:
        Les statistiques du batch inspecte.
    """
    ensure_dirs()
    spec = resolve_spec(dataset_name)
    effective_channels = _resolve_channels(spec, channels)

    train_loader, _ = get_image_dataloader(
        dataset_name,
        image_size,
        batch_size,
        channels=channels,
    )
    images, _ = next(iter(train_loader))
    stats = sanity_check_batch(
        images,
        expected_channels=effective_channels,
        expected_size=image_size,
    )

    if save_grid:
        grid_count = min(config.SAMPLE_GRID_SIZE, images.shape[0])
        destination = config.SAMPLES_DIR / f"{dataset_name}_real_{image_size}px.png"
        save_sample_grid(images[:grid_count], destination)
        stats["grid"] = str(destination)  # type: ignore[assignment]

    return stats


def main() -> None:
    """CLI : `python -m src.data.images [--dataset ...] [--no-grid]`."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset",
        choices=sorted(config.IMAGE_DATASETS),
        action="append",
        help="Dataset a preparer (repetable). Defaut : tous.",
    )
    parser.add_argument("--image-size", type=int, default=config.IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument(
        "--channels",
        type=int,
        choices=(1, 3),
        default=None,
        help="Force le nombre de canaux. Defaut : canaux natifs du dataset.",
    )
    parser.add_argument("--no-grid", action="store_true")
    args = parser.parse_args()

    for name in args.dataset or sorted(config.IMAGE_DATASETS):
        stats = prepare_dataset(
            name,
            args.image_size,
            batch_size=args.batch_size,
            channels=args.channels,
            save_grid=not args.no_grid,
        )
        formatted = ", ".join(
            f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in stats.items()
        )
        print(f"[ok] {name}: {formatted}")


if __name__ == "__main__":
    main()
