# ML Engineer — Étude de Stabilité DCGAN vs WGAN-GP
*Modélisation générative comparative et pipeline d'intégration Backend*

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)
---

## 🎯 Vue d'ensemble du projet

Ce dépôt contient l'implémentation de l'ingénierie Machine Learning pour l'étude comparative de stabilité entre le **DCGAN** (Deep Convolutional Generative Adversarial Network) et le **WGAN-GP** (Wasserstein GAN avec Gradient Penalty) appliquée à la génération d'images (**Fashion-MNIST**).

Les objectifs principaux de ce projet sont les suivants :
1. **Benchmark comparatif rigoureux :** Évaluation de la stabilité d'entraînement, de la vitesse de convergence, de la sensibilité au *mode collapse* et de la qualité des échantillons via la métrique **FID (Fréchet Inception Distance)** à travers des études d'ablation multi-seeds.
2. **Interopérabilité Backend prête pour la production :** Empaquetage des générateurs entraînés et des métadonnées sous forme d'artefacts conformes aux contrats de données pour une intégration fluide avec l'API FastAPI du Backend.

---

## 🚀 Statut actuel du projet

*   ✅ **Domaine image (Fashion-MNIST) :** Pipeline complet et fonctionnel couvrant le DCGAN, le WGAN-GP, le suivi d'expériences MLflow (backend SQLite), l'ablation multi-seeds, l'évaluation FID et l'export sécurisé des artefacts vers le Backend.
*   ⏸️ **Domaine tabulaire (Credit Card Fraud) :** Temporairement en pause selon la priorisation de l'équipe ; les structures de code de base et les contrats de données sont prêts pour une reprise ultérieure.

---

## 📂 Architecture du dépôt

```text
kougblenou-ml-engineer/
├── README.md                     <- Documentation complète du projet
├── notebooks/
│   └── dcgan_wgan_image.ipynb    <- Notebook principal du pipeline (exécuté sur GPU Google Colab)
├── src/
│   └── gan.py                    <- Architectures de modèles partagées (Generator / Discriminator)
└── .gitignore
```

---

## ⚙️ Points techniques clés

*   **Conformité stricte de l'architecture :** Implémentation des classes exactes `Generator` et `Discriminator` fournies dans `src/gan.py` pour garantir une compatibilité parfaite lors du chargement des poids en production (`torch.load`).
*   **Suivi robuste des expériences :** Intégration de **MLflow** avec un backend SQLite persistant (`mlflow.db`) pour suivre les hyperparamètres, les courbes de perte par époque, l'enregistrement des grilles d'échantillons visuels et le versioning des modèles.
*   **Ablation multi-seeds automatisée :** Pipelines scriptés s'exécutant sur plusieurs graines aléatoires pour détecter systématiquement les seuils de divergence et de *mode collapse*.
*   **Métriques d'évaluation quantitative :** Évaluation automatisée à l'aide de la **FID (Fréchet Inception Distance)** via `torch-fidelity` pour comparer le réalisme de la distribution générée par rapport aux données de validation réelles.
*   **Contrat d'empaquetage Backend :** Structure d'export standardisée (`<run_id>/generator.pt` + `config.json`) avec validation stricte du schéma avant livraison.

---

## 🛠️ Guide de démarrage rapide

### Prérequis
* Python 3.10+
* PyTorch 2.0+ avec support CUDA
* MLflow, Torchvision, TorchMetrics, Torch-Fidelity, Pandas, NumPy, Matplotlib

### Exécution via Google Colab
1. Téléchargez le dossier du projet dans votre Google Drive (`/MyDrive/Projet_Final_Master/`).
2. Ouvrez `notebooks/dcgan_wgan_image.ipynb` dans Google Colab.
3. Basculez le runtime sur **GPU** (`Exécution > Modifier le type d'exécution > GPU`).
4. Exécutez les cellules séquentiellement. Assurez-vous que `src/gan.py` est accessible via le chemin Python (`sys.path.append(...)`).

> ⏱️ **Temps d'exécution :** Environ 1 heure 45 minutes pour l'exécution complète (entraînement DCGAN + WGAN-GP, étude d'ablation sur 5 seeds et évaluation FID) sur un GPU T4 standard de Colab.

---

## 📦 Livraison des artefacts et intégration Backend

Les poids des modèles entraînés et leurs configurations sont structurés pour être consommés par le Backend :
* **Emplacement de stockage :** Volume partagé Google Drive `/exports_samples/` ou dépôt d'artefacts MLflow.
* **Format :**
  ```json
  {
    "run_id": "a1b2c3d4e5f6...",
    "model_type": "WGAN-GP",
    "latent_dim": 100,
    "image_channels": 1,
    "image_size": 28,
    "weights_path": "generator.pt"
  }
  ```

---

## 👥 Interfaces inter-équipes

| Membre / Rôle | Référence de documentation / Artefact | Objectif |
| :--- | :--- | :--- |
| **Data Engineer** | `docs/data_contract.md` | Schémas d'ingestion des données d'entrée et normes de normalisation |
| **Développeur Backend** | `docs/format_echange_backend.md` | Spécifications des checkpoints et contrats partagés (`src/gan.py`) |
| **Développeur Frontend** | Indirect via l'API Backend | Diffusion d'échantillons synthétiques générés à la demande |

---
