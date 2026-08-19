# Couche Data Engineering — Stabilité DCGAN vs WGAN-GP

Chaîne de préparation des données pour un projet de recherche comparant la
stabilité d'entraînement de **DCGAN** et **WGAN-GP**. Ce dépôt ne contient
**ni modèle GAN, ni boucle d'entraînement, ni application web** : uniquement
l'ingestion, le preprocessing, les splits, les garde-fous anti-fuite et la
plomberie de données de l'évaluation.

Trois sources, deux pipelines :

| Pipeline  | Sources                                               | Usage                                 |
| --------- | ----------------------------------------------------- | ------------------------------------- |
| Images    | Fashion-MNIST, CIFAR-10 (torchvision)                 | Entraînement GAN génératif         |
| Tabulaire | Credit Card Fraud (Kaggle`mlg-ulb/creditcardfraud`) | Augmentation de la classe minoritaire |

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

> **GPU.** `requirements.txt` installe la roue PyTorch par défaut, souvent une
> build CPU. Sur CPU, Inception traite ~6 images/s : le précalcul FID prend des
> dizaines de minutes et l'entraînement des GANs devient difficilement tenable.
> Avec un GPU NVIDIA, installer la build CUDA depuis
> [pytorch.org](https://pytorch.org/get-started/locally/) **avant** le reste.

---

## Démarrage rapide

```bash
python -m src.data.images
```

```bash
python -m src.data.tabular --eda
```

```bash
python -m src.data.fid_stats
```

```bash
pytest
```

La première commande télécharge Fashion-MNIST et CIFAR-10 et écrit une grille
d'échantillons ; la deuxième exige le CSV Kaggle (voir ci-dessous) ; la
troisième précalcule les statistiques FID de référence.

---

## Credentials Kaggle

Le dataset Credit Card Fraud n'est pas librement téléchargeable : il exige un
compte Kaggle. **Aucun credential n'est ni écrit dans le code, ni versionné.**
`kagglehub` et le CLI `kaggle` lisent eux-mêmes les identifiants ; le code de ce
dépôt ne fait que constater leur présence.

### Option A — fichier de token (recommandé)

1. Sur [kaggle.com/settings](https://www.kaggle.com/settings), cliquer
   **Create New Token**. Un `kaggle.json` est téléchargé.
2. Le déplacer vers :
   - Windows : `%USERPROFILE%\.kaggle\kaggle.json`
   - Linux/macOS : `~/.kaggle/kaggle.json` (puis `chmod 600 ~/.kaggle/kaggle.json`)

### Option B — variables d'environnement

```bash
export KAGGLE_USERNAME=votre_pseudo
```

```bash
export KAGGLE_KEY=votre_cle_api
```

Sous PowerShell : `$env:KAGGLE_USERNAME = "votre_pseudo"`.

### Option C — dépôt manuel

Sans compte utilisable, télécharger l'archive depuis
[kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
et extraire `creditcard.csv` dans `data/raw/`.

### Vérification

```bash
python -m src.data.tabular --download
```

En l'absence de credential, la commande sort en code 1 avec la marche à suivre
et le diagnostic de chaque tentative — pas de traceback.

> `kaggle.json` et `.env` sont exclus par `.gitignore`. Ne jamais les commiter.

---

## Étapes, une par une

### 1. Pipeline images

```bash
python -m src.data.images --dataset fashion_mnist --dataset cifar10
```

Télécharge vers `data/raw/`, valide un batch (plage, forme, canaux) et écrit
une grille de vérification visuelle dans `artifacts/samples/`.

Options utiles : `--image-size 32`, `--batch-size 128`, `--channels 3` (force
Fashion-MNIST en RGB pour partager une architecture entre les deux datasets),
`--no-grid`.

```python
from src.data.images import get_image_dataloader

train_loader, test_loader = get_image_dataloader("fashion_mnist", batch_size=128)
```

Les images sortent normalisées dans **[-1, 1]** — et non [0, 1] — car le
générateur se termine par une `tanh`. Fashion-MNIST est **paddé** de 28×28 à
32×32 plutôt que redimensionné : le padding préserve exactement les
statistiques de pixels, là où une interpolation introduirait un flou que le GAN
apprendrait comme une caractéristique des vraies images.

### 2. Pipeline tabulaire

```bash
python -m src.data.tabular --eda
```

Écrit dans `data/processed/` : matrices `.npy`, scaler `joblib` et métadonnées
JSON (empreinte de split incluse).

Options : `--seed 42`, `--test-size 0.2`, `--time-strategy drop|cyclical`.

```python
from src.data.tabular import preprocess_tabular, save_processed

split = preprocess_tabular(seed=42)
save_processed(split)

X_gan = split.minority_train   # fraudes du TRAIN : matériau du GAN d'augmentation
```

**L'ordre des opérations est imposé** et constitue la garantie anti-fuite :

1. split train/test **stratifié**, sur données réelles ;
2. `fit` du scaler sur le **train seul**, puis `transform` du test ;
3. isolation de la classe minoritaire **issue du train**.

`Time` est **droppée** par défaut : le dataset l'exprime en secondes depuis la
première transaction, soit un index d'acquisition. La conserver apprendrait au
GAN un artefact de collecte, et le split aléatoire en brise de toute façon
l'ordre. `--time-strategy cyclical` la ré-encode en heure de la journée (sin,
cos) si le rythme circadien vous intéresse. `Amount` passe par `log1p` avant
standardisation (asymétrie 15,4 → 0,34 sur le jeu réel).

### 3. Augmentation synthétique

```python
from src.data.augment import assert_no_synthetic_in_test, inject_synthetic, load_synthetic

generated = load_synthetic("dcgan_seed42", n_features=split.X_train.shape[1])
augmented = inject_synthetic(split.X_train, split.y_train, generated)

assert_no_synthetic_in_test(split.X_test, generated)   # le test reste 100 % réel
```

`inject_synthetic` ne reçoit **aucun jeu de test** : elle est structurellement
incapable de le contaminer. Les échantillons générés se déposent dans
`data/synthetic/` au format `.npy`. Les valeurs non finies (générateur divergé)
sont rejetées à l'injection.

`augmented.is_synthetic` est un masque booléen qui survit au mélange : l'origine
de chaque ligne reste auditable après coup.

### 4. Statistiques FID de référence

```bash
python -m src.data.fid_stats --n-samples 10000
```

Précalcule `(mu, sigma)` des activations Inception-v3 sur les **vraies** images
et met le résultat en cache dans `artifacts/fid_stats/`. Le côté réel du FID ne
change jamais : le recalculer à chaque évaluation, c'est repasser des dizaines
de milliers d'images dans Inception pour retrouver le même résultat.

```python
from src.data.fid_stats import get_or_compute_fid_statistics

stats = get_or_compute_fid_statistics("cifar10")   # cache si déjà calculé
mu, sigma = stats.mu, stats.sigma
```

Options : `--split train|test`, `--n-samples N`, `--force` (ignore le cache).

> **Taille d'échantillon.** `sigma` est une matrice 2048×2048 : l'estimer sur
> moins de ~2048 images la rend singulière. Le défaut de 10 000 est un
> compromis entre ce plancher et le coût CPU. Le FID absolu n'est alors pas
> comparable aux valeurs publiées (calculées sur le split entier), mais la
> comparaison DCGAN / WGAN-GP reste valide puisque les deux sont mesurés contre
> la même référence. `GAN_FID_NUM_SAMPLES=full` rétablit le split complet.

---

## Structure

```
data/
  raw/          # datasets bruts téléchargés          (non versionné)
  processed/    # arrays + scaler ajusté              (non versionné)
  synthetic/    # échantillons générés par les GANs   (non versionné)
artifacts/      # grilles d'échantillons, caches FID, index de splits (non versionné)
src/data/
  config.py     # SEED, chemins, constantes de preprocessing — source unique de vérité
  utils.py      # set_seed(), génération déterministe, hachage d'arrays
  images.py     # pipeline Fashion-MNIST / CIFAR-10
  tabular.py    # pipeline Credit Card Fraud
  splits.py     # splits déterministes réutilisables
  augment.py    # injection de synthétique, train uniquement
  fid_stats.py  # stats FID de référence + cache disque
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

### Prouver que deux runs ont vu les mêmes données

Indispensable avant de comparer DCGAN et WGAN-GP : un écart de FID n'est
attribuable à l'architecture que si le découpage était identique.

```python
from src.data.images import batch_order_fingerprint
from src.data.splits import assert_same_split
from src.data.tabular import verify_preprocessing_reproducibility

verify_preprocessing_reproducibility(frame, seed=42)          # tabulaire
assert_same_split(batch_order_fingerprint(a),                 # images
                  batch_order_fingerprint(b), context="cifar10")
```

### Surcharges par variable d'environnement

Pour s'adapter à une machine sans modifier le code :

| Variable                   | Défaut                 | Effet                                                  |
| -------------------------- | ----------------------- | ------------------------------------------------------ |
| `GAN_SEED`               | `42`                  | Seed globale                                           |
| `GAN_BATCH_SIZE`         | `128`                 | Taille de batch des DataLoaders                        |
| `GAN_NUM_WORKERS`        | `0` (Windows) / `4` | Workers de chargement                                  |
| `GAN_FID_BATCH_SIZE`     | `64`                  | Batch du calcul FID                                    |
| `GAN_FID_NUM_SAMPLES`    | `10000`               | Images de référence FID (`full` = split entier)    |
| `GAN_STRICT_DETERMINISM` | `0`                   | `1` = déterminisme strict (lève au lieu d'avertir) |

---

## Tests

```bash
pytest
```

```bash
pytest -m "not slow"
```

La suite passe sur une machine vierge : les tests dépendant d'un dataset non
téléchargé sont **skippés** avec un message actionnable, jamais en échec. Les
tests tabulaires s'appuient sur un jeu synthétique au même schéma et ne
requièrent donc aucun credential Kaggle.

| Fichier                     | Garantie vérifiée                                                  |
| --------------------------- | -------------------------------------------------------------------- |
| `test_setup.py`           | Cohérence de la config, seeding effectif                            |
| `test_images.py`          | Plage [-1, 1], formes, canaux par dataset                            |
| `test_splits.py`          | Stratification, disjonction, empreintes                              |
| `test_tabular.py`         | Le scaler n'a**jamais** vu le test ; split stratifié          |
| `test_leakage.py`         | Après augmentation, le test ne contient**aucun** synthétique |
| `test_reproducibility.py` | Même seed → splits et valeurs identiques                           |
| `test_fid_stats.py`       | `(mu, sigma)` exacts, clé de cache discriminante                  |
| `test_acceptance.py`      | Surface publique, absence de secret, exclusions git                  |

Chaque garantie de reproductibilité est doublée d'un **garde-fou anti
faux-positif** : un test qui ne saurait pas échouer ne prouverait rien.

---

## Périmètre

Ce dépôt couvre le rôle **Data Engineer**. Restent hors périmètre :

- architectures DCGAN / WGAN-GP, boucles d'entraînement, gradient penalty
  → ML / Research Engineer ;
- interprétation des métriques de stabilité → Data Scientist ;
- application web → Backend ;
- CI/CD, tracking d'expériences → MLOps.

Deux interfaces matérialisent la frontière : le **scaler persisté**, dont
`inverse_transform` ramène les échantillons générés dans l'espace des grandeurs
réelles, et `data/synthetic/`, point de dépôt validé des sorties du GAN.
