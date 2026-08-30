# Projet annuel — DCGAN vs WGAN-GP : étude de stabilité générative

Étude comparative de la stabilité d'entraînement entre DCGAN (Radford et al., 2015)
et WGAN-GP (Gulrajani et al., 2017) sur le dataset Fashion-MNIST, avec mesure de
la qualité (FID) et de la diversité des échantillons générés.

## Contenu de cette branche (backend)

Cette branche contient le backend FastAPI du projet : entraînement des modèles,
API de génération, calcul des métriques et tableau de stabilité.

```
app/
├── main.py            # Endpoints FastAPI (REST + WebSocket)
├── schemas.py         # Modèles Pydantic
├── training.py        # Boucle d'entraînement DCGAN/WGAN-GP, FID, diversité
├── model_contract.py  # Contrat de validation des modèles livrés
└── models/
    └── gan.py          # Architectures Generator / Discriminator
Dockerfile
docker-compose.yml
requirements.txt
```

## Installation locale

```bash
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Documentation interactive : `http://localhost:8000/docs`

## Avec Docker

```bash
docker compose up 
```

## Déploiement

API accessible en ligne : https://projet-annuel-93io.onrender.com/docs

## Reproductibilité

- Seeds fixés à l'entraînement (42, 123, 7)
- Architecture partagée (`app/models/gan.py`) entre l'entraînement et l'API
  pour garantir la cohérence des poids chargés
- Contrat strict de livraison de modèle (`generator.pt` + `config.json`),
  validé automatiquement à l'import

## Résultats principaux

| Métrique | DCGAN | WGAN-GP |
|---|---|---|
| Runs convergés (sur 3) | 2 | 3 |
| Mode collapse | 1 | 0 |
| FID moyen | 0.0345 | 0.0334 |
| Diversité moyenne | 0.7467 | 0.7538 |

Voir l'analyse critique complète dans le rapport joint.

## Équipe

- Data & ML Engineering  préparation des données et entraînement des modèles
- Backend & DevOps  API,conteneurisation, déploiement
- Frontend interface web de démonstration

=====================================================================
COMMANDES DE RÉIMPORT DES 6 MODÈLES SUR RENDER 
=====================================================================

À FAIRE : si le service Render s'est mis en veille (15 min d'inactivité),
il faut réimporter les 6 modèles

1) Vérifie d'abord que le service est réveillé :
   Ouvre https://projet-annuel-93io.onrender.com/docs dans un navigateur
   et attends que la page se charge (peut prendre 30-50 secondes).

2) Ouvre un terminal (Git Bash), place-vous dans le dossier projet-annuel :
   cd ~/Downloads/gan_backend/projet-annuel

3) Copie-colle les 6 commandes ci-dessous, une par une, dans le terminal.


--- Tes 6 modèles ---

curl -X POST https://projet-annuel-93io.onrender.com/api/models/upload \
  -F "generator_file=@data/models/0932eec0332f/generator.pt" \
  -F "config_file=@data/models/0932eec0332f/config.json" \
  -F "metrics_file=@data/models/0932eec0332f/metrics.json"

curl -X POST https://projet-annuel-93io.onrender.com/api/models/upload \
  -F "generator_file=@data/models/14ad111a11dc/generator.pt" \
  -F "config_file=@data/models/14ad111a11dc/config.json" \
  -F "metrics_file=@data/models/14ad111a11dc/metrics.json"

curl -X POST https://projet-annuel-93io.onrender.com/api/models/upload \
  -F "generator_file=@data/models/321bf5562ccd/generator.pt" \
  -F "config_file=@data/models/321bf5562ccd/config.json" \
  -F "metrics_file=@data/models/321bf5562ccd/metrics.json"

curl -X POST https://projet-annuel-93io.onrender.com/api/models/upload \
  -F "generator_file=@data/models/57e70ce22baf/generator.pt" \
  -F "config_file=@data/models/57e70ce22baf/config.json" \
  -F "metrics_file=@data/models/57e70ce22baf/metrics.json"

curl -X POST https://projet-annuel-93io.onrender.com/api/models/upload \
  -F "generator_file=@data/models/ddedd2ab570b/generator.pt" \
  -F "config_file=@data/models/ddedd2ab570b/config.json" \
  -F "metrics_file=@data/models/ddedd2ab570b/metrics.json"

curl -X POST https://projet-annuel-93io.onrender.com/api/models/upload \
  -F "generator_file=@data/models/eeef7ba1135a/generator.pt" \
  -F "config_file=@data/models/eeef7ba1135a/config.json" \
  -F "metrics_file=@data/models/eeef7ba1135a/metrics.json"

