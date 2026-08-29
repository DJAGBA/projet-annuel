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