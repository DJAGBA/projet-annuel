from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ModelType(str, Enum):
    dcgan = "dcgan"
    wgan_gp = "wgan_gp"


class DatasetName(str, Enum):
    fashion_mnist = "fashion_mnist"
    cifar10 = "cifar10"


class RunStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    diverged = "diverged"  # mode collapse / instabilité détectée


class CreateRunRequest(BaseModel):
    model_type: ModelType
    dataset: DatasetName = DatasetName.fashion_mnist
    epochs: int = Field(default=20, ge=1, le=200)
    batch_size: int = Field(default=64, ge=8, le=512)
    seed: int = Field(default=42, ge=0)
    latent_dim: int = Field(default=100, ge=8, le=512)
    lr: float = Field(default=2e-4, gt=0)
    n_critic: int = Field(default=5, ge=1, le=20, description="WGAN-GP uniquement")
    gp_lambda: float = Field(default=10.0, gt=0, description="WGAN-GP uniquement")


class RunSummary(BaseModel):
    run_id: str
    model_type: ModelType
    dataset: DatasetName
    seed: int
    status: RunStatus
    current_epoch: int
    total_epochs: int
    converged: Optional[bool] = None
    mode_collapse_detected: bool = False
    last_fid: Optional[float] = None
    last_diversity: Optional[float] = None


class LossPoint(BaseModel):
    step: int
    epoch: int
    d_loss: float
    g_loss: float
    gp_term: Optional[float] = None


class StabilityRow(BaseModel):
    model_type: ModelType
    dataset: DatasetName
    n_runs: int
    n_converged: int
    n_mode_collapse: int
    n_diverged: int
    avg_final_fid: Optional[float] = None
    avg_diversity: Optional[float] = None
    avg_wallclock_sec: Optional[float] = None