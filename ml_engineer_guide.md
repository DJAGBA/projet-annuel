# Guide de prise en main — ML Engineer

**Objet** : consommer la couche data sans dépendre d'un arbitrage préalable.
**Statut de la couche data** : livrée, testée (183 tests verts), opérationnelle.
**Public** : ML Engineer (DCGAN / WGAN-GP).

> Tous les extraits de ce document ont été **exécutés** contre la couche livrée
> avant d'y figurer. Les valeurs affichées sont des sorties réelles, pas des
> exemples inventés.

---

## Sommaire

1. [Ce dont tu disposes déjà](#1-ce-dont-tu-disposes-déjà)
2. [Mise en route (5 min)](#2-mise-en-route-5-min)
3. [Étape 1 — Charger les images](#3-étape-1--charger-les-images)
4. [Étape 2 — Calculer le FID de tes échantillons](#4-étape-2--calculer-le-fid-de-tes-échantillons)
5. [Étape 3 — Entraîner le GAN tabulaire](#5-étape-3--entraîner-le-gan-tabulaire)
6. [Étape 4 — Injecter le synthétique et évaluer](#6-étape-4--injecter-le-synthétique-et-évaluer)
7. [Étape 5 — Prouver que tes runs sont comparables](#7-étape-5--prouver-que-tes-runs-sont-comparables)
8. [Les 4 pièges, et comment les éviter](#8-les-4-pièges-et-comment-les-éviter)
9. [Référence rapide de l'API](#9-référence-rapide-de-lapi)

---

## 1. Ce dont tu disposes déjà

Tout est en place. **Rien ne te bloque, aucune décision n'est requise pour commencer.**

| Ressource | État | Emplacement |
|---|---|---|
| Fashion-MNIST + CIFAR-10, normalisés `[-1, 1]` | ✅ | via `get_image_dataloader()` |
| Statistiques FID de référence (2 datasets) | ✅ précalculées | `artifacts/fid_stats/*.npz` |
| Jeu tabulaire splitté, scalé, anti-fuite garanti | ✅ | `data/processed/*.npy` |
| 394 fraudes réelles isolées pour ton GAN | ✅ | `creditcard_X_minority_train.npy` |
| Scaler ajusté (pour `inverse_transform`) | ✅ | `creditcard_scaler.joblib` |
| Point de dépôt validé pour tes échantillons | ✅ | `data/synthetic/` |

### Volumes exacts

| Dataset | Train | Test | Shape d'un batch |
|---|---|---|---|
| Fashion-MNIST | 60 000 | 10 000 | `(B, 1, 32, 32)` `float32` |
| CIFAR-10 | 50 000 | 10 000 | `(B, 3, 32, 32)` `float32` |
| Credit Card — train | 227 845 dont **394 fraudes** | — | `(B, 29)` `float64` |
| Credit Card — test | — | 56 962 dont **98 fraudes** | `(B, 29)` `float64` |

---

## 2. Mise en route (5 min)

```bash
.venv\Scripts\activate
```

```bash
pytest -q
```

Attendu : `183 passed`. Si c'est vert, la couche data est saine et tu peux
enchaîner. Sinon, c'est un problème d'environnement, pas de ton code.

### Vérifier que les artefacts sont là

```python
from src.data.tabular import load_processed
from src.data.fid_stats import get_or_compute_fid_statistics

split = load_processed()                          # lève si data/processed/ est vide
stats = get_or_compute_fid_statistics("cifar10")  # lit le cache, ne recalcule pas
print(split.X_train.shape, stats.mu.shape)        # (227845, 29) (2048,)
```

Si `load_processed()` lève un `FileNotFoundError`, lance
`python -m src.data.tabular` — le message d'erreur te le dit explicitement.

### Toujours commencer par la seed

```python
from src.data.utils import set_seed

set_seed(42)   # AVANT toute opération GPU
```

`CUBLAS_WORKSPACE_CONFIG` doit être positionnée avant l'initialisation de cuBLAS.
Appelée trop tard, la fonction n'a plus d'effet sur le déterminisme CUDA.

---

## 3. Étape 1 — Charger les images

```python
from src.data.images import get_image_dataloader

train_loader, test_loader = get_image_dataloader(
    "cifar10",           # ou "fashion_mnist"
    image_size=32,
    batch_size=128,
    seed=42,
)

images, labels = next(iter(train_loader))
# images : torch.Size([128, 3, 32, 32])  float32  dans [-1, 1]
```

### Ce que tu reçois, garanti

- **Plage `[-1, 1]`**, pas `[0, 1]` → ton générateur se termine par `Tanh()`. ✅
- **`float32`**, prêt pour PyTorch, aucune conversion nécessaire.
- **Fashion-MNIST en 32×32**, paddé de 28×28 (bordure noire, pas d'interpolation).
- **Le test loader n'est ni mélangé ni tronqué** : identique à chaque évaluation,
  quelle que soit ta seed d'entraînement.

### Si tu veux une architecture unique pour les deux datasets

```python
# Fashion-MNIST en 3 canaux : le canal gris est répliqué
train_loader, _ = get_image_dataloader("fashion_mnist", channels=3)
# -> (B, 3, 32, 32)
```

### Si tu tiens absolument au 28×28

```python
train_loader, _ = get_image_dataloader("fashion_mnist", image_size=28)
# -> (B, 1, 28, 28), toujours dans [-1, 1]
```

**Mon conseil : reste en 32×32.** C'est une puissance de 2, donc une chaîne
d'*upsampling* propre `4 → 8 → 16 → 32` en convolutions transposées stride-2.
Avec 28, l'arithmétique kernel/stride/padding devient irrégulière et tu ne peux
plus partager l'architecture avec CIFAR-10.

### Visualiser tes échantillons générés

```python
from src.data.images import denormalize, save_sample_grid
from pathlib import Path

fake = generator(noise)                     # sortie tanh, dans [-1, 1]
save_sample_grid(fake[:64], Path("artifacts/samples/dcgan_epoch10.png"))
```

`save_sample_grid` dé-normalise et **clampe** : une sortie de générateur qui
déborde légèrement de `[-1, 1]` ne produira pas une image saturée trompeuse.

---

## 4. Étape 2 — Calculer le FID de tes échantillons

Le côté **réel** est déjà calculé et en cache. Tu n'as à calculer que le côté
**généré**, puis la distance entre les deux.

### 4.1 Le helper à copier dans ton code

```python
import numpy as np
from scipy import linalg


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Distance de Fréchet entre deux gaussiennes (= FID)."""
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if isinstance(covmean, tuple):          # compat. selon la version de SciPy
        covmean = covmean[0]
    if not np.isfinite(covmean).all():      # produit mal conditionné
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        if isinstance(covmean, tuple):
            covmean = covmean[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    value = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return float(max(value, 0.0))           # borne les -1e-9 dus au flottant
```

### 4.2 Calculer `(mu, sigma)` de tes images générées

```python
import torch
from torchmetrics.image.fid import FrechetInceptionDistance

from src.data.fid_stats import to_uint8_rgb
from src.data import config


def generated_statistics(fake_images, device="cpu", feature_dim=2048):
    """(mu, sigma) d'un lot d'images générées, dans [-1, 1]."""
    metric = FrechetInceptionDistance(feature=feature_dim, normalize=False).to(device)
    metric.eval()

    with torch.no_grad():
        for start in range(0, fake_images.shape[0], 64):
            batch = to_uint8_rgb(fake_images[start : start + 64]).to(device)
            metric.update(batch, real=True)      # 'real=True' : on veut juste les accumulateurs

    count = metric.real_features_num_samples
    mean = (metric.real_features_sum / count).unsqueeze(0)
    cov = (metric.real_features_cov_sum - count * mean.t().mm(mean)) / (count - 1)
    return mean.squeeze(0).double().cpu().numpy(), cov.double().cpu().numpy()
```

`to_uint8_rgb` fait pour toi les trois conversions qu'Inception exige :
`[-1, 1] → [0, 255]`, `float → uint8`, et `1 canal → 3 canaux` pour Fashion-MNIST.

### 4.3 Assembler

```python
from src.data.fid_stats import get_or_compute_fid_statistics

real = get_or_compute_fid_statistics("cifar10")        # cache : ~0,3 s
mu_fake, sigma_fake = generated_statistics(fake_images)

fid = frechet_distance(real.mu, real.sigma, mu_fake, sigma_fake)
print(f"FID = {fid:.2f}")
```

### 4.4 Repères mesurés, pour savoir si ton chiffre est plausible

| Comparaison | FID |
|---|---|
| Cache CIFAR-10 contre **lui-même** | **0,000** |
| CIFAR-10 train (10k) contre CIFAR-10 test (2k) | **16,4** |
| CIFAR-10 contre Fashion-MNIST | **187,3** |

Lis-les ainsi : **16,4 est le plancher de bruit d'échantillonnage** — deux
sous-ensembles des *mêmes* vraies images. Un GAN qui descend vers cet ordre de
grandeur est excellent. **187 est l'ordre de grandeur de deux distributions sans
rapport** — si ton GAN produit ça, il n'a rien appris.

> ⚠️ La référence porte sur **10 000 images**, pas sur le split complet. Tes FID
> sont donc comparables **entre eux** (DCGAN vs WGAN-GP vs seeds) mais **pas aux
> valeurs publiées** dans la littérature. Pour cela : `GAN_FID_NUM_SAMPLES=full`
> puis relancer `python -m src.data.fid_stats` (≈ 5 h sur CPU).

**Utilise toujours la même référence pour tous tes runs.** C'est ce qui rend la
comparaison DCGAN / WGAN-GP valide.

---

## 5. Étape 3 — Entraîner le GAN tabulaire

```python
import numpy as np
import torch
from src.data.tabular import load_processed

split = load_processed()

X_gan = split.minority_train.astype(np.float32)   # (394, 29) — voir note dtype
tensor = torch.from_numpy(X_gan)                  # torch.float32
```

### 🔴 Le point à ne pas rater : **pas de `Tanh()` sur le tabulaire**

Les features sont **standardisées z-score**, donc **non bornées**. Mesuré sur les
394 fraudes du train :

```
plage réelle : [-35.2, +37.0]
```

Un `Tanh()` en sortie ne produit que `[-1, 1]`. Ton générateur serait
**structurellement incapable** d'atteindre l'essentiel de l'espace des fraudes —
et **rien ne planterait** : l'entraînement convergerait vers des échantillons
tous tassés au centre, et le classifieur aval stagnerait sans cause visible.

```python
# ❌ à ne pas faire sur le tabulaire
self.out = nn.Sequential(nn.Linear(h, 29), nn.Tanh())

# ✅ sortie linéaire
self.out = nn.Linear(h, 29)
```

`Tanh` est une convention **image**, née du fait que les pixels sont bornés par
nature. Rien ne l'impose sur des features standardisées.

*(Si tu préfères vraiment garder `Tanh`, dis-le moi : je peux basculer le scaler
tabulaire en MinMax `[-1, 1]`. C'est une ligne de config, mais ça écrase les
queues de distribution — donc les fraudes extrêmes, les plus informatives.)*

### Revenir aux grandeurs réelles

```python
reels = split.scaler.inverse_transform(fake_scaled)   # (n, 29)

# Amount est en log1p : deux inversions successives
montants = np.expm1(reels[:, -1])
print(montants[:3].round(2))     # ex. [37.32  1.00  1.10] euros
```

Vérifié sur de vraies fraudes : `37,32 €`, `1,00 €`, `1,10 €` — des montants
plausibles, dont beaucoup de micro-transactions typiques du test de carte volée.

### Le schéma des 29 colonnes

| Position | Colonnes | Contenu |
|---|---|---|
| 0–27 | `V1` … `V28` | Composantes PCA anonymisées, standardisées |
| 28 | `Amount` | `log1p` puis standardisé |

`Time` a été retirée (index d'acquisition, pas un signal de fraude).
`split.feature_names` te redonne l'ordre exact à tout moment.

---

## 6. Étape 4 — Injecter le synthétique et évaluer

### 6.1 Déposer tes échantillons

```python
from src.data.augment import save_synthetic

save_synthetic(X_generated, "wgangp_seed42")     # -> data/synthetic/wgangp_seed42.npy
```

### 6.2 Injecter dans le train — jamais dans le test

```python
from src.data.augment import inject_synthetic, load_synthetic, assert_no_synthetic_in_test

generated = load_synthetic("wgangp_seed42", n_features=29)
augmented = inject_synthetic(split.X_train, split.y_train, generated, seed=42)

assert_no_synthetic_in_test(split.X_test, generated)   # garde-fou explicite
```

Résultat mesuré avec 1 000 échantillons générés :

```
train    227 845 -> 228 845
fraudes      394 ->   1 394
test     56 962 lignes, 100 % réel, inchangé
```

`inject_synthetic` **ne reçoit aucun jeu de test** : elle est structurellement
incapable de le contaminer. Elle **rejette aussi les NaN/inf** — si ton WGAN-GP
diverge, tu l'apprends à l'injection et non trois heures plus tard.

### 6.3 Le run de contrôle, à ne pas oublier

```python
control = inject_synthetic(split.X_train, split.y_train,
                           np.empty((0, 29)), seed=42)   # 0 % d'augmentation
```

C'est ta baseline. Elle est **mélangée exactement comme les runs augmentés**, pour
que l'ordre des lignes ne soit pas une variable qui diffère entre les protocoles.

### 6.4 Évaluer le classifieur aval

```python
clf.fit(augmented.X, augmented.y)          # train augmenté
score = clf.score(split.X_test, split.y_test)   # test 100 % réel, 98 fraudes
```

**Le test ne bouge jamais**, quel que soit le protocole. C'est ce qui rend les
scores « avec » et « sans » augmentation comparables.

### 6.5 Auditer après coup

```python
X_synth, y_synth = augmented.synthetic_subset()
X_real,  y_real  = augmented.real_subset()
print(augmented.summary())
# {'n_total': 228845, 'n_real': 227845, 'n_synthetic': 1000,
#  'synthetic_ratio': 0.00437, 'balance': {...}, 'seed': 42}
```

Le masque `augmented.is_synthetic` survit au mélange : l'origine de chaque ligne
reste traçable, ce qui serait perdu sans lui.

---

## 7. Étape 5 — Prouver que tes runs sont comparables

Un écart de FID entre DCGAN et WGAN-GP n'est imputable à l'architecture **que si
les deux ont vu exactement les mêmes données**. Sinon tu mesures aussi le hasard
du découpage.

### Images

```python
from src.data.images import batch_order_fingerprint
from src.data.splits import assert_same_split

loader_dcgan, _ = get_image_dataloader("cifar10", seed=42)
loader_wgan,  _ = get_image_dataloader("cifar10", seed=42)

assert_same_split(
    batch_order_fingerprint(loader_dcgan),
    batch_order_fingerprint(loader_wgan),
    context="cifar10",
)
# empreinte : 566b1f5d7ea8e3b0d4a1e5ca...
```

⚠️ `batch_order_fingerprint` **consomme** des batchs et fait avancer le générateur
du loader. Appelle-la sur un loader **neuf**, juste après construction.

### Tabulaire

```python
print(split.fingerprint)
# 553c09bc3d14c2745a3c86be725179b173b35f6b85192fb66f1d430e67d05609
```

**Reporte cette empreinte dans chacun de tes résultats.** C'est la preuve
opposable, en soutenance, que la comparaison porte sur des données identiques.

### Runs multi-seed

```python
from src.data import config
print(config.MULTI_SEEDS)   # (42, 1337, 2024)
```

Chaque seed donne un découpage différent (vérifié) et chaque seed est rejouable à
l'identique (vérifié). C'est ce qui te permet de rapporter une moyenne et un
écart-type de stabilité plutôt qu'un chiffre unique.

---

## 8. Les 4 pièges, et comment les éviter

| # | Piège | Symptôme | Solution |
|---|---|---|---|
| **1** | `Tanh()` sur le GAN tabulaire | Échantillons tassés au centre, classifieur qui stagne, **aucune erreur** | Sortie linéaire (§5) |
| **2** | `float64` tabulaire vs `float32` PyTorch | `RuntimeError: expected Float but found Double` au 1er forward | `.astype(np.float32)` (§5) |
| **3** | `set_seed()` appelée trop tard | Déterminisme CUDA silencieusement inopérant | L'appeler **en tout début de script** (§2) |
| **4** | `batch_order_fingerprint` sur un loader déjà itéré | Empreintes différentes alors que les données sont identiques | Loader neuf (§7) |

Le piège 1 est le seul qui ne produit **aucun signal d'erreur**. Les trois autres
échouent bruyamment, donc tu les verras.

### Deux points à trancher ensemble quand tu voudras

Ni l'un ni l'autre ne te bloque aujourd'hui :

- **Split de validation** — je livre train/test seulement. Utile si tu veux
  sélectionner un checkpoint ou régler les hyperparamètres du classifieur aval
  sans toucher au test. Coût faible de mon côté, dis-moi si tu en as besoin.
- **Version de scikit-learn** — le scaler est sérialisé par joblib et
  `requirements.txt` déclare `scikit-learn>=1.3` sans borne haute. J'ai constaté
  un `InconsistentVersionWarning` entre 1.9.0 (écriture) et 1.8.0 (lecture).
  À figer sur une version commune, sinon `inverse_transform` peut diverger d'un
  poste à l'autre.

---

## 9. Référence rapide de l'API

### Images — `src.data.images`

```python
get_image_dataloader(name, image_size=32, batch_size=128, *, channels=None,
                     num_workers=0, seed=42, download=True,
                     drop_last=None, shuffle_train=True)  -> (train, test)
denormalize(images)                    # [-1,1] -> [0,1], clampé
save_sample_grid(images, path, nrow=8)
sanity_check_batch(images, *, expected_channels, expected_size)
batch_order_fingerprint(loader, n_batches=3)  -> str
```

### Tabulaire — `src.data.tabular`

```python
load_processed(directory=...)  -> TabularSplit
preprocess_tabular(seed=42, *, test_size=0.2, time_strategy="drop") -> TabularSplit
```

`TabularSplit` : `.X_train` `.X_test` `.y_train` `.y_test` `.scaler`
`.feature_names` `.train_index` `.test_index` `.seed`
`.minority_train` *(propriété)* · `.fingerprint` *(propriété)* · `.summary()`

### Augmentation — `src.data.augment`

```python
inject_synthetic(X_train, y_train, X_synthetic, *, label=1, seed=42, shuffle=True)
save_synthetic(X, name)                     # -> data/synthetic/{name}.npy
load_synthetic(name, *, n_features=None)
assert_no_synthetic_in_test(X_test, X_synthetic)
assert_test_unchanged(expected, actual)
```

`AugmentedTrainingSet` : `.X` `.y` `.is_synthetic` `.n_real` `.n_synthetic`
`.synthetic_ratio` `.real_subset()` `.synthetic_subset()` `.summary()`

### FID — `src.data.fid_stats`

```python
get_or_compute_fid_statistics(name, *, split="train", n_samples=10000, force=False)
to_uint8_rgb(images)                   # [-1,1] float -> uint8 RGB
```

`FidStatistics` : `.mu` `.sigma` `.n_samples` `.dataset` `.split` `.feature_dim`

### Reproductibilité — `src.data.utils`, `src.data.splits`

```python
set_seed(42)                           # random + numpy + torch (CPU & CUDA)
assert_same_split(reference, candidate, *, context="")
verify_split_reproducibility(y, *, seed=42, n_runs=3)
hash_array(array) -> str
```

---

## En cas de blocage

1. `pytest -q` → si rouge, c'est l'environnement, pas ton code.
2. Les erreurs de la couche data sont **actionnables** : elles disent quoi lancer.
3. Le [README](readme.md) documente chaque commande ; le
   [data_contract.md](data_contract.md) porte les décisions ouvertes.
4. Sinon, viens me voir — c'est mon périmètre.
