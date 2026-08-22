import asyncio

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.model_contract import ModelContractError, load_generator
from app.schemas import CreateRunRequest, RunSummary, RunStatus, StabilityRow
from app.training import manager

MODELS_DIR = "data/models"  # dossier partagé (relatif au dossier où tu lances uvicorn) où le ML engineer / Colab dépose ses runs

app = FastAPI(title="GAN Stability Lab", version="1.0.0",
              description="DCGAN vs WGAN-GP : entraînement, génération et étude de stabilité")

# En dev, autorise le frontend local (à restreindre en prod).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _to_summary(state) -> RunSummary:
    return RunSummary(
        run_id=state.run_id,
        model_type=state.request.model_type,
        dataset=state.request.dataset,
        seed=state.request.seed,
        status=state.status,
        current_epoch=state.current_epoch,
        total_epochs=state.request.epochs,
        converged=state.converged,
        mode_collapse_detected=state.mode_collapse_detected,
        last_fid=state.last_fid,
        last_diversity=state.last_diversity,
    )


@app.post("/api/runs", response_model=RunSummary)
async def create_run(req: CreateRunRequest):
    """Crée et démarre un run d'entraînement DCGAN ou WGAN-GP en tâche de fond."""
    state = manager.create_run(req)
    asyncio.create_task(manager.run_training(state.run_id))
    return _to_summary(state)


@app.get("/api/runs", response_model=list[RunSummary])
async def list_runs():
    return [_to_summary(s) for s in manager.list_runs()]


@app.get("/api/runs/{run_id}", response_model=RunSummary)
async def get_run(run_id: str):
    state = manager.get_run(run_id)
    if state is None:
        raise HTTPException(404, "Run introuvable")
    return _to_summary(state)


@app.get("/api/runs/{run_id}/losses")
async def get_losses(run_id: str):
    """Historique complet des pertes (utile pour tracer la courbe après coup)."""
    state = manager.get_run(run_id)
    if state is None:
        raise HTTPException(404, "Run introuvable")
    return [p.model_dump() for p in state.losses]


@app.get("/api/runs/{run_id}/samples")
async def get_samples(run_id: str, n: int = 16):
    """Retourne une grille d'échantillons générés (PNG encodé en base64)."""
    state = manager.get_run(run_id)
    if state is None:
        raise HTTPException(404, "Run introuvable")
    if state.generator is None:
        raise HTTPException(409, "Le générateur n'est pas encore disponible pour ce run")
    grid_side = int(n ** 0.5)
    n = grid_side * grid_side  # force un carré parfait pour la grille
    b64 = manager.generate_grid_png_b64(run_id, n_samples=max(n, 4))
    return {"run_id": run_id, "image_base64": b64, "format": "png"}


@app.get("/api/stability", response_model=list[StabilityRow])
async def get_stability_table():
    """Tableau de stabilité DCGAN vs WGAN-GP (livrable requis de l'étude d'ablation)."""
    return manager.stability_table()


@app.websocket("/api/runs/{run_id}/ws")
async def run_websocket(websocket: WebSocket, run_id: str):
    """Diffuse les points de perte (et le statut final) en direct pendant l'entraînement."""
    state = manager.get_run(run_id)
    if state is None:
        await websocket.close(code=4404)
        return

    await websocket.accept()
    queue = await manager.subscribe(run_id)
    try:
        while True:
            payload = await queue.get()
            await websocket.send_json(payload)
            if payload.get("type") in ("done", "error"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        if queue in state.subscribers:
            state.subscribers.remove(queue)


@app.post("/api/runs/{run_id}/import", response_model=RunSummary)
async def import_external_model(run_id: str):
    """Charge un modèle livré par le ML engineer depuis /data/models/{run_id}/
    (generator.pt + config.json). Voir app/model_contract.py pour le schéma exact
    attendu dans config.json. Renvoie une erreur 422 explicite si le contrat
    n'est pas respecté (au lieu de planter plus tard à la génération)."""
    model_dir = f"{MODELS_DIR}/{run_id}"
    try:
        generator, config, metrics = load_generator(model_dir)
    except ModelContractError as exc:
        raise HTTPException(422, str(exc)) from exc

    # Enregistre le modèle importé comme un run "terminé" côté API,
    # pour qu'il soit utilisable par /samples et compté dans /stability.
    # metrics (fid/diversity) est optionnel : présent pour les runs Colab,
    # absent si le ML engineer ne fournit pas ce fichier hors-contrat.
    state = manager.import_run(run_id=config.run_id, model_type=config.model_type,
                                dataset=config.dataset, seed=config.seed,
                                epochs_trained=config.epochs_trained,
                                converged=config.converged,
                                mode_collapse_detected=config.mode_collapse_detected,
                                generator=generator, latent_dim=config.latent_dim,
                                last_fid=(metrics or {}).get("fid"),
                                last_diversity=(metrics or {}).get("diversity"))
    return _to_summary(state)


@app.get("/api/health")
async def health():
    return {"status": "ok"}