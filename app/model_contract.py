"""
Contrat d'interface entre le ML engineer et le backend.

Ce que le ML engineer doit livrer, pour chaque run entraîné, un dossier :

    models/{run_id}/
        generator.pt      <- torch.save(generator.state_dict(), ...)  [obligatoire]
        config.json        <- métadonnées décrites ci-dessous          [obligatoire]
        losses.json         <- historique des pertes (optionnel)

Le fichier config.json DOIT respecter exactement ce schéma (voir ModelConfig
ci-dessous). S'il ne correspond pas, load_generator() lève une erreur explicite
au chargement plutôt qu'un bug silencieux plus tard dans l'API.

Exemple de config.json valide :
{
  "run_id": "a1b2c3d4e5f6",
  "model_type": "wgan_gp",
  "dataset": "fashion_mnist",
  "latent_dim": 100,
  "channels": 1,
  "seed": 42,
  "epochs_trained": 40,
  "converged": true,
  "mode_collapse_detected": false
}
"""
import json
from pathlib import Path
from typing import Optional

import torch
from pydantic import BaseModel, ConfigDict, ValidationError

from app.models.gan import Generator
from app.schemas import DatasetName, ModelType


class ModelConfig(BaseModel):
    """Schéma strict attendu dans config.json. Tout champ manquant ou mal typé
    fait échouer la validation avec un message explicite (pas de bug silencieux)."""
    run_id: str
    model_type: ModelType
    dataset: DatasetName
    latent_dim: int
    channels: int
    seed: int
    epochs_trained: int
    converged: Optional[bool] = None
    mode_collapse_detected: bool = False

    model_config = ConfigDict(extra="forbid")  # rejette tout champ inattendu -> force la conformité


class ModelContractError(Exception):
    """Levée quand ce que fournit le ML engineer ne respecte pas le contrat."""


def load_generator(model_dir: str | Path) -> tuple[Generator, ModelConfig, Optional[dict]]:
    """Charge un générateur entraîné à partir du dossier livré par le ML engineer
    (ou d'un run Colab suivant le même format).

    Lève ModelContractError avec un message clair si :
      - config.json est absent ou n'est pas un JSON valide
      - un champ du contrat manque ou a le mauvais type
      - generator.pt est absent ou incompatible avec la config déclarée

    Retourne aussi metrics (dict ou None) : contenu de metrics.json s'il existe
    (fichier optionnel, hors contrat strict — utilisé par le script Colab pour
    transmettre fid/diversity/wallclock_sec, mais absent si le ML engineer ne
    le fournit pas).
    """
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    weights_path = model_dir / "generator.pt"
    metrics_path = model_dir / "metrics.json"

    if not config_path.exists():
        raise ModelContractError(f"config.json manquant dans {model_dir}")
    if not weights_path.exists():
        raise ModelContractError(f"generator.pt manquant dans {model_dir}")

    try:
        raw = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        raise ModelContractError(f"config.json n'est pas un JSON valide : {exc}") from exc

    try:
        config = ModelConfig(**raw)
    except ValidationError as exc:
        raise ModelContractError(
            f"config.json ne respecte pas le contrat attendu :\n{exc}"
        ) from exc

    generator = Generator(latent_dim=config.latent_dim, channels=config.channels)
    try:
        state_dict = torch.load(weights_path, map_location="cpu")
        generator.load_state_dict(state_dict)
    except Exception as exc:  # noqa: BLE001
        raise ModelContractError(
            f"generator.pt incompatible avec latent_dim={config.latent_dim}, "
            f"channels={config.channels} déclarés dans config.json : {exc}"
        ) from exc

    generator.eval()

    metrics = None
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            metrics = None  # optionnel : on ignore si mal formé, pas bloquant

    return generator, config, metrics