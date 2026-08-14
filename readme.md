# Couche Data Engineering — Stabilité DCGAN vs WGAN-GP

Chaîne de préparation des données pour un projet de recherche comparant la
stabilité d'entraînement de **DCGAN** et **WGAN-GP**. Ce dépôt ne contient
**ni modèle GAN, ni boucle d'entraînement, ni application web** : uniquement
l'ingestion, le preprocessing, les splits, les garde-fous anti-fuite et la
plomberie de données de l'évaluation.

Trois sources, deux pipelines :

| Pipeline | Sources | Usage |
|---|---|---|
| Images | Fashion-MNIST, CIFAR-10 (torchvision) | Entraînement GAN génératif |
| Tabulaire | Credit Card Fraud (Kaggle `mlg-ulb/creditcardfraud`) | Augmentation de la classe minoritaire |

---

## Installation

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

Python 3.10+ requis. Sous Linux/macOS, l'activation est `source .venv/bin/activate`.

---

## Structure

```
data/
  raw/          # datasets bruts téléchargés      (non versionné)
  processed/    # arrays + scaler ajusté          (non versionné)
  synthetic/    # échantillons générés par les GANs (non versionné)
artifacts/       # grilles d'échantillons, caches FID, index de splits (non versionné)
src/data/
  config.py     # SEED, chemins, constantes de preprocessing — source unique de vérité
  utils.py      # set_seed(), génération déterministe, hachage d'arrays
  images.py     # pipeline Fashion-MNIST / CIFAR-10          (phase 1)
  tabular.py    # pipeline Credit Card Fraud                 (phase 2)
  splits.py     # splits déterministes réutilisables         (phase 2)
  augment.py    # injection de synthétique, train uniquement (phase 3)
  fid_stats.py  # stats FID de référence + cache disque      (phase 4)
tests/
```

---

## Reproductibilité

Toute la configuration vit dans [`src/data/config.py`](src/data/config.py) —
seed, chemins, tailles d'image, ratio de split, stratégie de traitement de
`Time`. Aucune constante de preprocessing ne doit être écrite ailleurs.

`set_seed()` couvre `random`, `numpy`, `torch` (CPU **et** CUDA), `PYTHONHASHSEED`
et les backends cuDNN/cuBLAS. Deux points d'attention :

- **Appeler `set_seed()` en tout début de script**, avant toute opération GPU :
  `CUBLAS_WORKSPACE_CONFIG` doit être positionnée avant l'initialisation de cuBLAS.
- **Le déterminisme est non strict par défaut.** Plusieurs *backwards* de
  convolution transposée — utilisés par les générateurs DCGAN — n'ont pas
  d'implémentation déterministe CUDA et feraient échouer l'entraînement en mode
  strict. Passer `GAN_STRICT_DETERMINISM=1` pour lever une exception au lieu
  d'un avertissement.

Avec `num_workers > 0`, le seeding du processus principal ne suffit pas :
utiliser `seed_worker` en `worker_init_fn` et `make_generator()` pour le
`generator` du `DataLoader` (les deux sont câblés par la factory du pipeline
images).

### Surcharges par variable d'environnement

Pour s'adapter à une machine sans modifier le code :

| Variable | Défaut | Effet |
|---|---|---|
| `GAN_SEED` | `42` | Seed globale |
| `GAN_BATCH_SIZE` | `128` | Taille de batch des DataLoaders |
| `GAN_NUM_WORKERS` | `0` (Windows) / `4` | Workers de chargement |
| `GAN_FID_BATCH_SIZE` | `64` | Batch du calcul FID |
| `GAN_STRICT_DETERMINISM` | `0` | `1` = déterminisme strict (lève au lieu d'avertir) |

---

## Tests

```bash
pytest
```

La suite passe sur une machine vierge : les tests dépendant d'un dataset non
téléchargé sont **skippés** avec un message actionnable, jamais en échec.

---

## Secrets

Aucun credential n'est présent dans le code ni versionné. `kaggle.json` et
`.env` sont exclus par `.gitignore`. La procédure d'authentification Kaggle est
documentée à la phase 2 (pipeline tabulaire).
