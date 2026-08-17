"""Configuration centralisee de la couche data engineering.

Ce module ne contient **aucune logique** et n'importe **aucune dependance
lourde** (ni torch, ni pandas) : il doit rester importable instantanement,
y compris depuis les tests. Toute constante de preprocessing, tout chemin et
toute seed du projet vivent ici et nulle part ailleurs.

Certaines valeurs peuvent etre surchargees par variable d'environnement
(prefixe ``GAN_``) pour s'adapter a la machine sans modifier le code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Reproductibilite
# ---------------------------------------------------------------------------

#: Seed unique du projet. Toute source d'alea (splits, shuffle, init) en derive.
SEED: int = int(os.environ.get("GAN_SEED", 42))

#: Seeds utilisees pour les runs multi-seed (robustesse des comparaisons).
MULTI_SEEDS: tuple[int, ...] = (42, 1337, 2024)

#: `torch.use_deterministic_algorithms(True, warn_only=STRICT_DETERMINISM is False)`.
#: En mode strict, PyTorch leve une exception sur toute operation sans
#: implementation deterministe (certains backwards de conv transposee, utilises
#: par les generateurs DCGAN, sont dans ce cas) -> desactive par defaut.
STRICT_DETERMINISM: bool = os.environ.get("GAN_STRICT_DETERMINISM", "0") == "1"


# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------

#: Racine du depot (ce fichier est a <root>/src/data/config.py).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW: Path = DATA_DIR / "raw"
DATA_PROCESSED: Path = DATA_DIR / "processed"
DATA_SYNTHETIC: Path = DATA_DIR / "synthetic"

#: Sorties non-dataset : grilles d'echantillons, caches FID, index de splits.
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
SAMPLES_DIR: Path = ARTIFACTS_DIR / "samples"
FID_CACHE_DIR: Path = ARTIFACTS_DIR / "fid_stats"
SPLITS_DIR: Path = ARTIFACTS_DIR / "splits"

#: Repertoires crees a la demande par `utils.ensure_dirs()`.
MANAGED_DIRS: tuple[Path, ...] = (
    DATA_RAW,
    DATA_PROCESSED,
    DATA_SYNTHETIC,
    SAMPLES_DIR,
    FID_CACHE_DIR,
    SPLITS_DIR,
)


# ---------------------------------------------------------------------------
# Pipeline images
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageDatasetSpec:
    """Description d'un dataset image.

    Centraliser ces attributs permet d'ecrire **un seul** pipeline de
    transforms parametrable au lieu d'une branche par dataset.

    Attributes:
        name: Cle publique utilisee par la factory (`get_image_dataloader`).
        torchvision_class: Nom de la classe dans `torchvision.datasets`.
        channels: Nombre de canaux natifs (1 pour Fashion-MNIST, 3 pour CIFAR-10).
        native_size: Cote de l'image brute en pixels (28 pour FMNIST, 32 pour CIFAR).
        num_classes: Nombre de classes (informatif, pour les GANs conditionnels).
    """

    name: str
    torchvision_class: str
    channels: int
    native_size: int
    num_classes: int

    @property
    def norm_mean(self) -> tuple[float, ...]:
        """Moyenne de normalisation, un element par canal."""
        return (0.5,) * self.channels

    @property
    def norm_std(self) -> tuple[float, ...]:
        """Ecart-type de normalisation, un element par canal.

        Avec mean=std=0.5, `Normalize` envoie [0, 1] -> [-1, 1], ce qu'exige
        la sortie `tanh` du generateur.
        """
        return (0.5,) * self.channels


IMAGE_DATASETS: dict[str, ImageDatasetSpec] = {
    "fashion_mnist": ImageDatasetSpec(
        name="fashion_mnist",
        torchvision_class="FashionMNIST",
        channels=1,
        native_size=28,
        num_classes=10,
    ),
    "cifar10": ImageDatasetSpec(
        name="cifar10",
        torchvision_class="CIFAR10",
        channels=3,
        native_size=32,
        num_classes=10,
    ),
}

#: Cote cible commun aux deux datasets : Fashion-MNIST est padde de 28 -> 32.
IMAGE_SIZE: int = 32

#: Bornes attendues apres normalisation (verifiees par les sanity checks).
PIXEL_RANGE: tuple[float, float] = (-1.0, 1.0)

BATCH_SIZE: int = int(os.environ.get("GAN_BATCH_SIZE", 128))

#: 0 par defaut sous Windows : le spawn des workers y est couteux et casse
#: souvent l'execution depuis un notebook / pytest.
NUM_WORKERS: int = int(os.environ.get("GAN_NUM_WORKERS", 0 if os.name == "nt" else 4))

DROP_LAST: bool = True  # batchs de taille constante : evite les stats de BN bruitees
SAMPLE_GRID_SIZE: int = 64  # nb d'images de la grille de verification visuelle


# ---------------------------------------------------------------------------
# Pipeline tabulaire (Credit Card Fraud)
# ---------------------------------------------------------------------------

#: Identifiant du dataset sur Kaggle (aucun credential ici, cf. README).
CREDITCARD_KAGGLE_ID: str = "mlg-ulb/creditcardfraud"
CREDITCARD_CSV_NAME: str = "creditcard.csv"
CREDITCARD_CSV: Path = DATA_RAW / CREDITCARD_CSV_NAME

TIME_COL: str = "Time"
AMOUNT_COL: str = "Amount"
TARGET_COL: str = "Class"
V_COLUMNS: tuple[str, ...] = tuple(f"V{i}" for i in range(1, 29))

#: Etiquette de la classe minoritaire (fraudes) = celle a augmenter par GAN.
FRAUD_LABEL: int = 1

#: Proportion du jeu de test, split stratifie sur `Class`.
TEST_SIZE: float = 0.2

#: Sort de la colonne `Time`.
#: - "drop" (defaut) : `Time` est le nombre de secondes ecoulees depuis la
#:   premiere transaction du dataset, donc un simple index d'acquisition. La
#:   conserver apprendrait au GAN un artefact de collecte, pas un signal de
#:   fraude, et introduirait un ordre temporel que le split aleatoire brise.
#: - "cyclical" : re-encodage en heure de la journee (sin/cos), qui garde le
#:   rythme circadien sans la derive absolue.
TIME_STRATEGY: Literal["drop", "cyclical"] = "drop"
SECONDS_PER_DAY: int = 86_400

#: `Amount` est fortement dissymetrique (queue lourde) : log1p avant
#: standardisation, sinon le scaler est domine par quelques valeurs extremes.
AMOUNT_LOG1P: bool = True

# Noms de fichiers produits dans data/processed/
PROCESSED_PREFIX: str = "creditcard"
SCALER_FILENAME: str = f"{PROCESSED_PREFIX}_scaler.joblib"
PROCESSED_METADATA: str = f"{PROCESSED_PREFIX}_metadata.json"


# ---------------------------------------------------------------------------
# Statistiques FID de reference
# ---------------------------------------------------------------------------

#: Couche Inception-v3 utilisee (2048 = pool3, la convention de la litterature).
FID_FEATURE_DIM: int = 2048
FID_BATCH_SIZE: int = int(os.environ.get("GAN_FID_BATCH_SIZE", 64))

#: Nombre d'images reelles servant de reference FID.
#:
#: 10 000 est un compromis assume. Deux bornes l'encadrent :
#: - plancher : `sigma` est une matrice `FID_FEATURE_DIM x FID_FEATURE_DIM`
#:   (2048^2). L'estimer sur moins de ~2048 images la rend singuliere et le FID
#:   devient numeriquement instable ;
#: - plafond : sur CPU, Inception traite ~6 images/s, soit ~5 h pour les splits
#:   complets des deux datasets.
#:
#: Le FID absolu obtenu n'est pas comparable aux valeurs publiees (calculees sur
#: le split entier), mais la comparaison DCGAN / WGAN-GP reste valide : les deux
#: sont mesures contre cette meme reference.
#:
#: `GAN_FID_NUM_SAMPLES=full` (ou `none`) retablit le split complet.
_FID_SAMPLES_ENV: str = os.environ.get("GAN_FID_NUM_SAMPLES", "10000")
FID_NUM_SAMPLES: int | None = (
    None if _FID_SAMPLES_ENV.lower() in ("none", "full", "0") else int(_FID_SAMPLES_ENV)
)

FID_SPLIT: Literal["train", "test"] = "train"
