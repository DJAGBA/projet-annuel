import asyncio
import base64
import io
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.models.gan import Discriminator, Generator, gradient_penalty, weights_init
from app.schemas import CreateRunRequest, LossPoint, ModelType, RunStatus, StabilityRow
from torchmetrics.image.fid import FrechetInceptionDistance

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/tmp/gan_data"

# Seuils heuristiques pour la détection d'instabilité (utilisés pour le tableau
# de stabilité DCGAN vs WGAN-GP demandé dans les livrables).
D_LOSS_SATURATION_THRESHOLD = 1e-3  # D domine totalement -> signe de mode collapse (DCGAN)
LOSS_EXPLOSION_THRESHOLD = 1e4      # divergence numérique
STD_WINDOW = 50                     # fenêtre glissante pour juger de l'oscillation
FID_N_SAMPLES = 512                 # nombre d'images réelles/générées comparées pour le FID


@dataclass
class RunState:
    run_id: str
    request: CreateRunRequest
    status: RunStatus = RunStatus.pending
    current_epoch: int = 0
    losses: List[LossPoint] = field(default_factory=list)
    converged: Optional[bool] = None
    mode_collapse_detected: bool = False
    last_fid: Optional[float] = None
    last_diversity: Optional[float] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    subscribers: List[asyncio.Queue] = field(default_factory=list)
    generator: Optional[nn.Module] = None
    latent_dim: int = 100
    channels: int = 1


class TrainingManager:
    def __init__(self):
        self.runs: Dict[str, RunState] = {}

    # ---------- lifecycle ----------

    def create_run(self, req: CreateRunRequest) -> RunState:
        run_id = uuid.uuid4().hex[:12]
        state = RunState(run_id=run_id, request=req, latent_dim=req.latent_dim,
                          channels=1 if req.dataset.value == "fashion_mnist" else 3)
        self.runs[run_id] = state
        return state

    def import_run(self, run_id: str, model_type, dataset, seed: int, epochs_trained: int,
                    converged: Optional[bool], mode_collapse_detected: bool,
                    generator, latent_dim: int, last_fid: Optional[float] = None,
                    last_diversity: Optional[float] = None) -> RunState:
        """Enregistre un modèle livré par le ML engineer (hors entraînement via l'API)
        comme un run consultable normalement par /samples, /stability, etc."""
        fake_req = CreateRunRequest(model_type=model_type, dataset=dataset, seed=seed,
                                     epochs=max(epochs_trained, 1), latent_dim=latent_dim)
        state = RunState(run_id=run_id, request=fake_req, latent_dim=latent_dim,
                          channels=1 if dataset.value == "fashion_mnist" else 3)
        state.status = RunStatus.completed
        state.current_epoch = epochs_trained
        state.converged = converged
        state.mode_collapse_detected = mode_collapse_detected
        state.generator = generator
        state.last_fid = last_fid
        state.last_diversity = last_diversity
        self.runs[run_id] = state
        return state

    def get_run(self, run_id: str) -> Optional[RunState]:
        return self.runs.get(run_id)

    def list_runs(self) -> List[RunState]:
        return list(self.runs.values())

    async def subscribe(self, run_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self.runs[run_id].subscribers.append(q)
        return q

    async def _broadcast(self, state: RunState, payload: dict) -> None:
        for q in state.subscribers:
            await q.put(payload)

    # ---------- data ----------

    def _get_dataloader(self, req: CreateRunRequest) -> DataLoader:
        tfm = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * (1 if req.dataset.value == "fashion_mnist" else 3),
                                  [0.5] * (1 if req.dataset.value == "fashion_mnist" else 3)),
        ])
        if req.dataset.value == "fashion_mnist":
            ds = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=tfm)
        else:
            ds = datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=tfm)
        return DataLoader(ds, batch_size=req.batch_size, shuffle=True, drop_last=True, num_workers=2)

    # ---------- FID (qualité de génération, livrable requis) ----------

    @torch.no_grad()
    def _compute_fid(self, generator: nn.Module, loader: DataLoader, latent_dim: int,
                      channels: int) -> float:
        """Calcule le FID entre un échantillon d'images réelles du dataset et des
        images générées. torchmetrics attend des images uint8 [0,255] à 3 canaux."""
        fid = FrechetInceptionDistance(feature=64, normalize=False).to(DEVICE)
        generator.eval()

        def to_uint8_rgb(x: torch.Tensor) -> torch.Tensor:
            x = (x.clamp(-1, 1) + 1) / 2 * 255  # [-1,1] -> [0,255]
            x = x.to(torch.uint8)
            if channels == 1:
                x = x.repeat(1, 3, 1, 1)  # niveaux de gris -> RGB (requis par Inception)
            return x

        n_collected = 0
        for real, _ in loader:
            fid.update(to_uint8_rgb(real).to(DEVICE), real=True)
            n_collected += real.size(0)
            if n_collected >= FID_N_SAMPLES:
                break

        n_fake = 0
        while n_fake < FID_N_SAMPLES:
            batch = min(64, FID_N_SAMPLES - n_fake)
            z = torch.randn(batch, latent_dim, device=DEVICE)
            fake = generator(z)
            fid.update(to_uint8_rgb(fake), real=False)
            n_fake += batch

        generator.train()
        return float(fid.compute().item())

    @torch.no_grad()
    def _compute_diversity(self, generator: nn.Module, latent_dim: int,
                            n_samples: int = 64) -> float:
        """Diversité des échantillons générés (livrable requis : "mesurer la
        diversité des échantillons"). Approche simple : distance L2 moyenne entre
        toutes les paires d'images générées, en espace pixel normalisé. Un score
        bas indique que le générateur produit des images très similaires entre
        elles, signe classique de mode collapse partiel (même si le FID reste bon)."""
        generator.eval()
        z = torch.randn(n_samples, latent_dim, device=DEVICE)
        imgs = generator(z)  # (N, C, H, W), valeurs dans [-1, 1]
        generator.train()

        flat = imgs.view(n_samples, -1)  # (N, C*H*W)
        # Distance L2 moyenne sur toutes les paires (i != j), normalisée par la
        # dimension du vecteur pour rester comparable entre tailles d'image.
        dists = torch.cdist(flat, flat, p=2)  # (N, N)
        mask = ~torch.eye(n_samples, dtype=torch.bool, device=DEVICE)
        avg_dist = dists[mask].mean().item()
        return avg_dist / (flat.shape[1] ** 0.5)

    # ---------- main entrypoint (run as background task) ----------

    async def run_training(self, run_id: str) -> None:
        state = self.runs[run_id]
        req = state.request
        state.status = RunStatus.running
        state.started_at = time.time()
        torch.manual_seed(req.seed)

        try:
            loader = await asyncio.to_thread(self._get_dataloader, req)
            channels = state.channels

            G = Generator(latent_dim=req.latent_dim, channels=channels).to(DEVICE)
            is_wgan = req.model_type == ModelType.wgan_gp
            D = Discriminator(channels=channels, use_batchnorm=not is_wgan,
                               use_sigmoid=not is_wgan).to(DEVICE)
            G.apply(weights_init)
            D.apply(weights_init)
            state.generator = G

            if is_wgan:
                opt_g = optim.Adam(G.parameters(), lr=req.lr, betas=(0.0, 0.9))
                opt_d = optim.Adam(D.parameters(), lr=req.lr, betas=(0.0, 0.9))
            else:
                opt_g = optim.Adam(G.parameters(), lr=req.lr, betas=(0.5, 0.999))
                opt_d = optim.Adam(D.parameters(), lr=req.lr, betas=(0.5, 0.999))

            bce = nn.BCELoss()
            step = 0
            recent_g_losses: List[float] = []
            diverged = False

            for epoch in range(req.epochs):
                state.current_epoch = epoch + 1
                for real, _ in loader:
                    real = real.to(DEVICE)
                    bsz = real.size(0)
                    gp_term = None

                    if is_wgan:
                        # --- n_critic pas de critic pour 1 pas de générateur ---
                        for _ in range(req.n_critic):
                            z = torch.randn(bsz, req.latent_dim, device=DEVICE)
                            fake = G(z).detach()
                            opt_d.zero_grad()
                            gp = gradient_penalty(D, real, fake, DEVICE)
                            d_loss = D(fake).mean() - D(real).mean() + req.gp_lambda * gp
                            d_loss.backward()
                            opt_d.step()
                        gp_term = gp.item()

                        z = torch.randn(bsz, req.latent_dim, device=DEVICE)
                        fake = G(z)
                        opt_g.zero_grad()
                        g_loss = -D(fake).mean()
                        g_loss.backward()
                        opt_g.step()

                        d_loss_val, g_loss_val = d_loss.item(), g_loss.item()
                    else:
                        # --- DCGAN: BCE classique ---
                        z = torch.randn(bsz, req.latent_dim, device=DEVICE)
                        fake = G(z)
                        real_labels = torch.ones(bsz, device=DEVICE)
                        fake_labels = torch.zeros(bsz, device=DEVICE)

                        opt_d.zero_grad()
                        d_loss_real = bce(D(real), real_labels)
                        d_loss_fake = bce(D(fake.detach()), fake_labels)
                        d_loss = d_loss_real + d_loss_fake
                        d_loss.backward()
                        opt_d.step()

                        opt_g.zero_grad()
                        g_loss = bce(D(fake), real_labels)
                        g_loss.backward()
                        opt_g.step()

                        d_loss_val, g_loss_val = d_loss.item(), g_loss.item()

                    point = LossPoint(step=step, epoch=epoch, d_loss=d_loss_val,
                                       g_loss=g_loss_val, gp_term=gp_term)
                    state.losses.append(point)
                    await self._broadcast(state, {"type": "loss", "data": point.model_dump()})

                    recent_g_losses.append(g_loss_val)
                    if len(recent_g_losses) > STD_WINDOW:
                        recent_g_losses.pop(0)

                    if abs(d_loss_val) > LOSS_EXPLOSION_THRESHOLD or abs(g_loss_val) > LOSS_EXPLOSION_THRESHOLD:
                        diverged = True
                        break
                    if not is_wgan and d_loss_val < D_LOSS_SATURATION_THRESHOLD:
                        state.mode_collapse_detected = True

                    step += 1
                    await asyncio.sleep(0)  # laisse la boucle event tourner (websocket, etc.)

                if diverged:
                    break

            state.finished_at = time.time()
            if diverged:
                state.status = RunStatus.diverged
                state.converged = False
            else:
                state.status = RunStatus.completed
                state.converged = not state.mode_collapse_detected
                # FID calculé seulement si le run n'a pas divergé (mesure de qualité
                # requise dans les livrables : "Mesurer la qualité de génération (FID...)").
                try:
                    state.last_fid = await asyncio.to_thread(
                        self._compute_fid, G, loader, req.latent_dim, channels
                    )
                    await self._broadcast(state, {"type": "fid", "value": state.last_fid})
                except Exception as fid_exc:  # noqa: BLE001
                    # Le FID est une métrique de qualité, pas un critère bloquant :
                    # si son calcul échoue, le run reste "completed" mais last_fid=None.
                    await self._broadcast(state, {"type": "fid_error", "message": str(fid_exc)})

                # Diversité des échantillons (livrable requis, complémentaire au FID :
                # le FID mesure la ressemblance au vrai dataset, la diversité mesure
                # si le générateur ne produit pas toujours des variantes du même échantillon).
                try:
                    state.last_diversity = await asyncio.to_thread(
                        self._compute_diversity, G, req.latent_dim
                    )
                    await self._broadcast(state, {"type": "diversity", "value": state.last_diversity})
                except Exception as div_exc:  # noqa: BLE001
                    await self._broadcast(state, {"type": "diversity_error", "message": str(div_exc)})

            await self._broadcast(state, {"type": "done", "status": state.status.value})

        except Exception as exc:  # noqa: BLE001
            state.status = RunStatus.failed
            state.finished_at = time.time()
            await self._broadcast(state, {"type": "error", "message": str(exc)})
            raise

    # ---------- sampling ----------

    @torch.no_grad()
    def generate_grid_png_b64(self, run_id: str, n_samples: int = 16) -> str:
        state = self.runs[run_id]
        if state.generator is None:
            raise ValueError("Aucun générateur entraîné pour ce run")
        G = state.generator
        G.eval()
        z = torch.randn(n_samples, state.latent_dim, device=DEVICE)
        imgs = G(z).cpu()
        imgs = (imgs + 1) / 2  # [-1,1] -> [0,1]
        G.train()

        grid_side = int(n_samples ** 0.5)
        _, c, h, w = imgs.shape
        canvas = Image.new("RGB" if c == 3 else "L", (w * grid_side, h * grid_side))
        for i in range(grid_side * grid_side):
            img = imgs[i].permute(1, 2, 0).numpy()
            img = (img * 255).astype("uint8")
            if c == 1:
                tile = Image.fromarray(img.squeeze(-1), mode="L")
            else:
                tile = Image.fromarray(img, mode="RGB")
            canvas.paste(tile, ((i % grid_side) * w, (i // grid_side) * h))

        buf = io.BytesIO()
        canvas.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ---------- stability table (livrable requis) ----------

    def stability_table(self) -> List[StabilityRow]:
        groups: Dict[tuple, List[RunState]] = {}
        for s in self.runs.values():
            key = (s.request.model_type, s.request.dataset)
            groups.setdefault(key, []).append(s)

        rows = []
        for (model_type, dataset), states in groups.items():
            finished = [s for s in states if s.status in (RunStatus.completed, RunStatus.diverged)]
            n_converged = sum(1 for s in finished if s.converged)
            n_collapse = sum(1 for s in finished if s.mode_collapse_detected)
            n_diverged = sum(1 for s in finished if s.status == RunStatus.diverged)
            fids = [s.last_fid for s in finished if s.last_fid is not None]
            diversities = [s.last_diversity for s in finished if s.last_diversity is not None]
            times = [s.finished_at - s.started_at for s in finished
                     if s.finished_at and s.started_at]
            rows.append(StabilityRow(
                model_type=model_type,
                dataset=dataset,
                n_runs=len(states),
                n_converged=n_converged,
                n_mode_collapse=n_collapse,
                n_diverged=n_diverged,
                avg_final_fid=sum(fids) / len(fids) if fids else None,
                avg_diversity=sum(diversities) / len(diversities) if diversities else None,
                avg_wallclock_sec=sum(times) / len(times) if times else None,
            ))
        return rows


manager = TrainingManager()