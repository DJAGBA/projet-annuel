# Image de base légère avec Python
FROM python:3.11-slim

WORKDIR /app

# Installer les dépendances système nécessaires à torch/torchvision
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python d'abord (meilleure mise en cache Docker :
# si seul le code change, cette étape lente n'est pas refaite à chaque build)
# PyTorch en version CPU uniquement (pas de GPU dans le conteneur de déploiement :
# l'entraînement se fait ailleurs, ex. Google Colab ; ce backend ne fait que
# charger des modèles déjà entraînés et servir l'API). Réduit énormément la
# taille du téléchargement/build par rapport à la version CUDA par défaut.
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application
COPY app/ ./app/

# Dossier où seront déposés les modèles importés (ML engineer / Colab)
RUN mkdir -p /app/data/models

EXPOSE 8000

# --host 0.0.0.0 est indispensable en conteneur : 127.0.0.1 ne serait
# accessible que depuis l'intérieur du conteneur lui-même.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]