"""Statistiques FID de reference, precalculees sur les **vraies** images.

Le FID compare deux gaussiennes ajustees sur les activations Inception-v3 :
celle des images reelles et celle des images generees. Le cote reel ne change
jamais au cours d'un projet -- le recalculer a chaque evaluation, c'est repasser
60 000 images dans Inception pour retrouver le meme resultat.

Ce module calcule une fois `(mu, sigma)` par dataset et les met en cache disque.
Deux benefices, l'un pratique et l'autre scientifique : les evaluations
suivantes sont immediates, et **DCGAN et WGAN-GP sont compares a exactement la
meme reference**, y compris quand le calcul n'utilise qu'un sous-echantillon.

Perimetre : ce module produit la reference reelle. Le calcul du FID des images
generees releve de l'evaluation, pas de la couche data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data import config
from src.data.images import denormalize, get_image_datasets, resolve_spec
from src.data.utils import ensure_dirs, get_device, resolve_pin_memory

__all__ = [
    "FidStatistics",
    "to_uint8_rgb",
    "fid_cache_path",
    "compute_real_fid_statistics",
    "get_or_compute_fid_statistics",
]


# ---------------------------------------------------------------------------
# Conversion vers le format attendu par Inception
# ---------------------------------------------------------------------------


def to_uint8_rgb(images: torch.Tensor) -> torch.Tensor:
    """Convertit un batch du pipeline vers le format attendu par Inception-v3.

    Le pipeline produit du float dans [-1, 1] et, pour Fashion-MNIST, un seul
    canal. Inception attend du `uint8` RGB. La replication du canal unique sur
    trois est la convention retenue par la litterature pour le FID sur images
    en niveaux de gris : elle laisse le reseau voir une image achromatique
    plutot qu'une image coloree arbitrairement.

    Args:
        images: Batch `(B, C, H, W)` normalise dans [-1, 1], `C` valant 1 ou 3.

    Returns:
        Un batch `(B, 3, H, W)` en `uint8`, valeurs dans [0, 255].

    Raises:
        ValueError: Si le nombre de canaux n'est ni 1 ni 3.
    """
    channels = images.shape[1]
    if channels not in (1, 3):
        raise ValueError(f"1 ou 3 canaux attendus, recu {channels}.")

    unit_range = denormalize(images)
    if channels == 1:
        unit_range = unit_range.repeat(1, 3, 1, 1)
    return unit_range.mul(255.0).round().clamp(0, 255).to(torch.uint8)


# ---------------------------------------------------------------------------
# Statistiques
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FidStatistics:
    """Reference gaussienne des activations Inception sur images reelles.

    Attributes:
        mu: Moyenne des activations, de forme `(feature_dim,)`.
        sigma: Covariance des activations, `(feature_dim, feature_dim)`.
        n_samples: Nombre d'images ayant servi au calcul.
        dataset: Nom du dataset source.
        split: Split utilise (`"train"` ou `"test"`).
        image_size: Cote des images en entree du pipeline.
        feature_dim: Dimension des activations (2048 = pool3).
        seed: Graine du sous-echantillonnage, `None` si le split est complet.
    """

    mu: np.ndarray
    sigma: np.ndarray
    n_samples: int
    dataset: str
    split: str
    image_size: int
    feature_dim: int
    seed: int | None

    def metadata(self) -> dict[str, object]:
        """Descripteur serialisable en JSON, sans les matrices."""
        return {
            "dataset": self.dataset,
            "split": self.split,
            "n_samples": self.n_samples,
            "image_size": self.image_size,
            "feature_dim": self.feature_dim,
            "seed": self.seed,
        }

    def save(self, path: Path) -> Path:
        """Ecrit les statistiques dans un `.npz` compresse.

        Args:
            path: Fichier de destination.

        Returns:
            Le chemin ecrit.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            mu=self.mu,
            sigma=self.sigma,
            metadata=np.asarray(json.dumps(self.metadata())),
        )
        return path

    @classmethod
    def load(cls, path: Path) -> FidStatistics:
        """Recharge des statistiques mises en cache.

        Args:
            path: Fichier `.npz` produit par `save`.

        Returns:
            Les `FidStatistics` reconstituees.

        Raises:
            FileNotFoundError: Si le cache n'existe pas.
        """
        if not path.exists():
            raise FileNotFoundError(f"Aucune statistique FID en cache a {path}.")

        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"]))
            return cls(
                mu=payload["mu"],
                sigma=payload["sigma"],
                n_samples=int(metadata["n_samples"]),
                dataset=str(metadata["dataset"]),
                split=str(metadata["split"]),
                image_size=int(metadata["image_size"]),
                feature_dim=int(metadata["feature_dim"]),
                seed=metadata["seed"],
            )


def fid_cache_path(
    dataset_name: str,
    *,
    split: str = config.FID_SPLIT,
    n_samples: int | None = config.FID_NUM_SAMPLES,
    image_size: int = config.IMAGE_SIZE,
    feature_dim: int = config.FID_FEATURE_DIM,
    seed: int = config.SEED,
    directory: Path = config.FID_CACHE_DIR,
) -> Path:
    """Construit le chemin de cache d'un jeu de statistiques.

    Tout parametre qui change le resultat entre dans le nom du fichier. Sans
    cela, un cache calcule sur 5 000 images serait silencieusement reutilise
    pour une evaluation censee porter sur le split complet -- et le FID
    obtenu ne serait comparable a rien.

    Args:
        dataset_name: Cle de `config.IMAGE_DATASETS`.
        split: `"train"` ou `"test"`.
        n_samples: Taille du sous-echantillon, `None` pour tout le split.
        image_size: Cote des images.
        feature_dim: Dimension des activations.
        seed: Graine du sous-echantillonnage (ignoree si `n_samples is None`).
        directory: Repertoire de cache.

    Returns:
        Le chemin du `.npz` correspondant.
    """
    count = "full" if n_samples is None else f"n{n_samples}"
    suffix = "" if n_samples is None else f"_seed{seed}"
    name = f"{dataset_name}_{split}_{count}_{image_size}px_f{feature_dim}{suffix}.npz"
    return directory / name


# ---------------------------------------------------------------------------
# Calcul
# ---------------------------------------------------------------------------


def _statistics_dataloader(
    dataset_name: str,
    *,
    split: str,
    image_size: int,
    batch_size: int,
    n_samples: int | None,
    seed: int,
    num_workers: int,
    download: bool,
) -> tuple[DataLoader, int]:
    """Construit un loader deterministe, non melange, pour le calcul des stats.

    Volontairement independant de `get_image_dataloader` : le loader
    d'entrainement melange ses batchs et ferait avancer son generateur. Ici on
    veut une passe unique, ordonnee et sans effet de bord.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split doit valoir 'train' ou 'test', recu {split!r}.")

    train_ds, test_ds = get_image_datasets(
        dataset_name, image_size, download=download
    )
    dataset = train_ds if split == "train" else test_ds
    total = len(dataset)  # type: ignore[arg-type]

    if n_samples is not None and n_samples < total:
        # Sous-echantillon tire par permutation seedee plutot que par troncature :
        # prendre les n premieres images epouserait l'ordre du fichier source,
        # qui n'a aucune raison d'etre representatif.
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(total, generator=generator)[:n_samples].tolist()
        dataset = Subset(dataset, indices)
        total = n_samples

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=resolve_pin_memory(),
    )
    return loader, total


def _real_statistics_from_metric(metric) -> tuple[np.ndarray, np.ndarray]:
    """Extrait `(mu, sigma)` des accumulateurs d'une metrique torchmetrics.

    Reprend l'arithmetique exacte de `FrechetInceptionDistance.compute()` :
    les statistiques mises en cache sont ainsi celles que la metrique aurait
    utilisees elle-meme, et non une variante recalculee autrement.

    Raises:
        AttributeError: Si torchmetrics change le nom de ses accumulateurs.
    """
    required = ("real_features_sum", "real_features_cov_sum", "real_features_num_samples")
    missing = [name for name in required if not hasattr(metric, name)]
    if missing:
        raise AttributeError(
            f"Accumulateurs torchmetrics introuvables : {missing}. "
            "La version installee a probablement change son etat interne."
        )

    count = metric.real_features_num_samples
    mean = (metric.real_features_sum / count).unsqueeze(0)
    cov_num = metric.real_features_cov_sum - count * mean.t().mm(mean)
    cov = cov_num / (count - 1)

    return (
        mean.squeeze(0).double().cpu().numpy(),
        cov.double().cpu().numpy(),
    )


def compute_real_fid_statistics(
    dataset_name: str,
    *,
    split: Literal["train", "test"] = config.FID_SPLIT,
    n_samples: int | None = config.FID_NUM_SAMPLES,
    image_size: int = config.IMAGE_SIZE,
    feature_dim: int = config.FID_FEATURE_DIM,
    batch_size: int = config.FID_BATCH_SIZE,
    seed: int = config.SEED,
    device: torch.device | None = None,
    num_workers: int = config.NUM_WORKERS,
    download: bool = True,
    progress: bool = True,
) -> FidStatistics:
    """Calcule `(mu, sigma)` sur les vraies images d'un dataset.

    Args:
        dataset_name: Cle de `config.IMAGE_DATASETS`.
        split: Split de reference.
        n_samples: Nombre d'images. `None` prend tout le split. Sur CPU, un
            sous-echantillon de quelques milliers d'images reduit fortement le
            temps de calcul, au prix d'un biais du FID -- biais sans effet sur
            la comparaison DCGAN / WGAN-GP tant que la meme reference sert aux
            deux.
        image_size: Cote des images en entree du pipeline.
        feature_dim: Dimension des activations Inception.
        batch_size: Taille de batch pour la passe Inception.
        seed: Graine du sous-echantillonnage.
        device: Device de calcul. `None` choisit CUDA si disponible.
        num_workers: Process de chargement.
        download: Telecharge le dataset si absent.
        progress: Affiche l'avancement.

    Returns:
        Les `FidStatistics` calculees.

    Raises:
        ImportError: Si `torchmetrics[image]` n'est pas installe.
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError as exc:  # pragma: no cover - depend de l'environnement
        raise ImportError(
            "torchmetrics[image] est requis pour les statistiques FID. "
            'Installer avec : pip install "torchmetrics[image]>=1.2"'
        ) from exc

    resolve_spec(dataset_name)  # valide le nom avant tout travail couteux
    device = device or get_device()

    loader, total = _statistics_dataloader(
        dataset_name,
        split=split,
        image_size=image_size,
        batch_size=batch_size,
        n_samples=n_samples,
        seed=seed,
        num_workers=num_workers,
        download=download,
    )

    metric = FrechetInceptionDistance(feature=feature_dim, normalize=False).to(device)
    metric.eval()

    seen = 0
    with torch.no_grad():
        for images, _ in loader:
            batch = to_uint8_rgb(images).to(device)
            metric.update(batch, real=True)
            seen += batch.shape[0]
            if progress:
                print(
                    f"\r  {dataset_name}/{split} : {seen:,}/{total:,} images",
                    end="",
                    flush=True,
                )
    if progress:
        print()

    mu, sigma = _real_statistics_from_metric(metric)
    return FidStatistics(
        mu=mu,
        sigma=sigma,
        n_samples=seen,
        dataset=dataset_name,
        split=split,
        image_size=image_size,
        feature_dim=feature_dim,
        seed=None if n_samples is None else seed,
    )


def get_or_compute_fid_statistics(
    dataset_name: str,
    *,
    split: Literal["train", "test"] = config.FID_SPLIT,
    n_samples: int | None = config.FID_NUM_SAMPLES,
    image_size: int = config.IMAGE_SIZE,
    feature_dim: int = config.FID_FEATURE_DIM,
    seed: int = config.SEED,
    directory: Path = config.FID_CACHE_DIR,
    force: bool = False,
    **kwargs: object,
) -> FidStatistics:
    """Retourne les statistiques depuis le cache, ou les calcule puis les cache.

    Args:
        dataset_name: Cle de `config.IMAGE_DATASETS`.
        split: Split de reference.
        n_samples: Nombre d'images, `None` pour tout le split.
        image_size: Cote des images.
        feature_dim: Dimension des activations.
        seed: Graine du sous-echantillonnage.
        directory: Repertoire de cache.
        force: Recalcule meme si un cache existe.
        **kwargs: Transmis a `compute_real_fid_statistics`.

    Returns:
        Les `FidStatistics`, chargees ou fraichement calculees.
    """
    path = fid_cache_path(
        dataset_name,
        split=split,
        n_samples=n_samples,
        image_size=image_size,
        feature_dim=feature_dim,
        seed=seed,
        directory=directory,
    )

    if path.exists() and not force:
        return FidStatistics.load(path)

    statistics = compute_real_fid_statistics(
        dataset_name,
        split=split,
        n_samples=n_samples,
        image_size=image_size,
        feature_dim=feature_dim,
        seed=seed,
        **kwargs,  # type: ignore[arg-type]
    )
    statistics.save(path)
    return statistics


def main() -> None:
    """CLI : `python -m src.data.fid_stats [--dataset ...] [--n-samples N]`."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset",
        choices=sorted(config.IMAGE_DATASETS),
        action="append",
        help="Dataset a traiter (repetable). Defaut : tous.",
    )
    parser.add_argument("--split", choices=("train", "test"), default=config.FID_SPLIT)
    parser.add_argument(
        "--n-samples",
        type=int,
        default=config.FID_NUM_SAMPLES,
        help="Nombre d'images. Omettre pour tout le split (long sur CPU).",
    )
    parser.add_argument("--batch-size", type=int, default=config.FID_BATCH_SIZE)
    parser.add_argument("--force", action="store_true", help="Ignore le cache existant.")
    args = parser.parse_args()

    ensure_dirs()
    for name in args.dataset or sorted(config.IMAGE_DATASETS):
        statistics = get_or_compute_fid_statistics(
            name,
            split=args.split,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            force=args.force,
        )
        path = fid_cache_path(
            name, split=args.split, n_samples=args.n_samples
        )
        print(
            f"[ok] {name}/{args.split} : n={statistics.n_samples:,} "
            f"mu{statistics.mu.shape} sigma{statistics.sigma.shape} -> {path.name}"
        )


if __name__ == "__main__":
    main()
