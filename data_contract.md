# Data Contract — Interface Data Engineer → ML Engineer

**Projet** : GAN pour génération/augmentation de données — Étude de stabilité (DCGAN vs WGAN-GP)
**Entre** : Data Engineer (fournisseur) et ML Engineer (consommateur)
**Statut** : ☑ Brouillon — **côté Data Engineer rempli**, en attente d'arbitrage ML Engineer sur §6
**Dernière mise à jour** : 2026-08-19

> Ce document définit précisément ce que le Data Engineer doit fournir au ML Engineer,
> et ce que le ML Engineer attend en retour, pour éviter toute ambiguïté ou incohérence
> silencieuse entre le pipeline de données et le pipeline d'entraînement.

> **⚠️ À lire en premier : §6 recense 3 conflits ouverts entre ce qui est livré et ce que
> le gabarit pré-remplissait en §5. Le conflit C1 casserait l'entraînement tabulaire sans
> lever d'erreur.**

---

## 1. Datasets retenus

Accord sur les deux catégories en parallèle. **Les deux sont livrés et opérationnels** ;
la priorité « image d'abord » n'a pas eu à jouer.

| Dataset | Rôle dans le projet | Statut de livraison |
|---|---|---|
| **Image** (Fashion-MNIST + CIFAR-10) | Étude de stabilité principale — DCGAN vs WGAN-GP sur plusieurs seeds | ✅ Livré |
| **Tabulaire** (Credit Card Fraud, `mlg-ulb/creditcardfraud`) | Évaluation de l'augmentation sur cas d'usage réel | ✅ Livré |

---

## 2. Dataset image

### 2.1 Format des fichiers

| Champ | Valeur livrée |
|---|---|
| Format de stockage | **Pas de fichier statique.** Livré via une factory Python : `get_image_dataloader(name, image_size, batch_size, ...) -> (train_loader, test_loader)`. Les datasets bruts torchvision restent dans `data/raw/`. |
| Résolution | **32×32 pour les deux** (Fashion-MNIST paddé de 28→32). Paramétrable : `image_size=28` redonne 28×28. → cf. conflit **C2** |
| Canaux | 1 (Fashion-MNIST) / 3 (CIFAR-10), **natifs par défaut**. Paramétrable via `channels=1|3` dans les deux sens. |
| Normalisation | **[-1, 1]**, déjà appliquée par le DE. Le ML Engineer reçoit du normalisé, jamais du brut. |
| Type de données | `torch.float32` (images), `torch.int64` (labels) |

**Vérifié à l'exécution** :

```
fashion_mnist  batch (B, 1, 32, 32)  float32  min -1.0000  max 1.0000
cifar10        batch (B, 3, 32, 32)  float32  min -1.0000  max 1.0000
```

Le padding 28→32 est un **padding noir** (valeur 0 → -1 après normalisation), pas un
redimensionnement : les statistiques de pixels sont préservées à l'identique, là où une
interpolation introduirait un flou que le discriminateur apprendrait comme signature du réel.

### 2.2 Structure attendue

| Question | Réponse DE |
|---|---|
| Split train / validation / test | **train / test uniquement** (splits natifs torchvision). **Pas de split de validation.** → cf. conflit **C4** |
| Convention de nommage | Sans objet (API Python, pas de fichiers). Clés de dataset : `"fashion_mnist"`, `"cifar10"`. |
| Emplacement | Local, `data/raw/` (non versionné, retéléchargeable). Pas de S3 ni DVC. |

Le **test loader n'est ni mélangé ni tronqué** (`shuffle=False`, `drop_last=False`) : il est
parcouru identiquement à chaque évaluation, quelle que soit la seed d'entraînement. C'est ce
qui permet de comparer des FID entre runs.

### 2.3 Volume

| Dataset | Train | Test |
|---|---|---|
| Fashion-MNIST | 60 000 | 10 000 |
| CIFAR-10 | 50 000 | 10 000 |

- **Suffisant pour 5 seeds d'ablation** : chaque seed rejoue le même dataset complet, le
  volume ne se divise pas entre seeds.
- **Dataset figé** : torchvision vérifie l'intégrité par MD5 au téléchargement. Aucune
  évolution possible en cours de projet.
- **Preuve d'identité entre runs** : `batch_order_fingerprint(loader)` retourne un SHA-256
  de la séquence de batchs servie. Deux runs à même seed donnent la même empreinte.

---

## 3. Dataset tabulaire déséquilibré

### 3.1 Format des fichiers

| Champ | Valeur livrée |
|---|---|
| Format | **`.npy` (NumPy), pas CSV ni Parquet.** Le CSV brut reste dans `data/raw/creditcard.csv` si besoin. |
| Encodage | Sans objet (binaire) |
| Séparateur | Sans objet |

Artefacts dans `data/processed/` :

| Fichier | Contenu | Shape | dtype |
|---|---|---|---|
| `creditcard_X_train.npy` | Features train scalées | (227 845, 29) | `float64` |
| `creditcard_X_test.npy` | Features test scalées | (56 962, 29) | `float64` |
| `creditcard_y_train.npy` | Labels train | (227 845,) | `int64` |
| `creditcard_y_test.npy` | Labels test | (56 962,) | `int64` |
| `creditcard_X_minority_train.npy` | **Fraudes du train — matériau du GAN** | (394, 29) | `float64` |
| `creditcard_train_index.npy` / `_test_index.npy` | Indices dans le CSV source | — | `int64` |
| `creditcard_scaler.joblib` | `StandardScaler` ajusté **sur le train seul** | — | — |
| `creditcard_metadata.json` | Seed, noms de features, empreinte, balances | — | — |

Rechargement en un appel : `load_processed()` → objet `TabularSplit`.

→ dtype `float64` : cf. conflit **C3**.

### 3.2 Schéma des colonnes

**29 features**, dans cet ordre exact :

| Position | Colonnes | Type | Description | Contraintes |
|---|---|---|---|---|
| 0–27 | `V1` … `V28` | `float64` | Composantes PCA anonymisées (fournies telles quelles par la source) | Aucune valeur manquante |
| 28 | `Amount` | `float64` | Montant, **`log1p` puis standardisé** | Aucune valeur manquante |
| — | *label séparé* | `int64` | `0` = légitime, `1` = fraude | Aucune valeur manquante |

**Le label n'est pas une colonne** : il est livré dans un array `y` distinct. La colonne
source s'appelle `Class` (et non `is_fraud`) ; le renommage est sans objet côté `.npy`.

**`Time` est retirée.** Le dataset l'exprime en secondes depuis la première transaction,
soit un index d'acquisition : la conserver apprendrait au GAN un artefact de collecte, et
le split aléatoire en brise de toute façon l'ordre. Un ré-encodage cyclique
(`time_sin`, `time_cos`, → 31 features) est implémenté et testé, désactivé par défaut —
activable par `--time-strategy cyclical` si le ML Engineer le souhaite.

### 3.3 Nettoyage déjà effectué par le Data Engineer

- ☑ **Valeurs manquantes** : aucune dans la source ; vérifié, aucun traitement nécessaire.
- ☑ **Outliers** : **non supprimés.** `Amount` est transformé par `log1p` avant
  standardisation (asymétrie **16,98 → 0,163** sur le jeu réel). Aucune ligne n'est écartée :
  supprimer des extrêmes retirerait des fraudes, précisément ce que le GAN doit apprendre.
- ☑ **Colonnes catégorielles** : aucune. `V1`–`V28` sont déjà des sorties PCA numériques.
- ☑ **Standardisation** : **appliquée**, `StandardScaler` (z-score) ajusté sur le train seul.

### 3.4 Gestion du déséquilibre des classes

- **Taux de déséquilibre observé** : **0,1727 %** de fraudes (492 sur 284 807), soit
  **1 fraude pour 578 transactions**.
- ☑ **Aucun rééquilibrage appliqué en amont.** Ni SMOTE, ni undersampling, ni class weights.
  Le ratio natif est préservé à l'identique dans le train et le test. **Aucun risque de
  double traitement.**
- ☐ Rééquilibrage partiel — *non applicable*.

| Sous-ensemble | Lignes | Fraudes | Ratio |
|---|---|---|---|
| Train | 227 845 | **394** | 0,17292 % |
| Test | 56 962 | **98** | 0,17204 % |

### 3.5 Split train/val/test

- **Méthode** : `train_test_split` **stratifié sur le label**, `test_size=0.2`,
  `random_state=42` passé explicitement (indépendant de l'état global du RNG).
  **Pas de split de validation** → cf. conflit **C4**.
- **Empreinte du split** (à citer dans tout run pour prouver l'identité des données) :

```
553c09bc3d14c2745a3c86be725179b173b35f6b85192fb66f1d430e67d05609
```

- ☑ **Vérification anti-fuite — confirmée et sous test automatisé.**

  L'ordre est imposé : **(1)** split stratifié d'abord, sur données réelles → **(2)** `fit`
  du scaler sur le train seul, `transform` du test → **(3)** isolation de la classe
  minoritaire issue du train.

  Preuves opposables, rejouables par `pytest` :

  | Vérification | Constat |
  |---|---|
  | `scaler.n_samples_seen_` | **227 845** = taille du train exactement, pas 284 807 |
  | Moyenne du train après scaling | **0 exactement** (`atol=1e-10`) — impossible si le fit avait vu tout le jeu |
  | Moyenne du test après scaling | **≠ 0** — le test est transformé, jamais ajusté |
  | Intersection train ∩ test | **vide** (indices) |

---

## 4. Pipeline de livraison

| Question | Réponse DE |
|---|---|
| Fichier statique ou pipeline reproductible ? | **Pipeline reproductible par scripts.** `python -m src.data.images`, `python -m src.data.tabular`, `python -m src.data.fid_stats`. Pas de DVC ni Airflow. Les artefacts sont régénérables à l'identique depuis la seed. |
| Notification en cas de changement des sources | **Via git.** Les sources sont figées (MD5 torchvision, dataset Kaggle versionné `3`). Tout changement de convention passe par une révision de ce contrat (§7). |
| Versioning des données | **Empreinte SHA-256 du split** (`553c09bc…`), stockée dans `creditcard_metadata.json` et vérifiable par `assert_same_split()`. Les données elles-mêmes ne sont pas versionnées (exclues par `.gitignore`) : c'est la **recette** qui l'est. |

### Contrôle d'identité avant toute comparaison DCGAN / WGAN-GP

```python
from src.data.splits import assert_same_split
from src.data.images import batch_order_fingerprint
from src.data.tabular import verify_preprocessing_reproducibility

verify_preprocessing_reproducibility(frame, seed=42)                      # tabulaire
assert_same_split(batch_order_fingerprint(a), batch_order_fingerprint(b)) # images
```

Un écart de FID n'est imputable à l'architecture que si ces contrôles passent.

### Dépôt des échantillons générés (sens ML Engineer → DE)

```python
from src.data.augment import save_synthetic, load_synthetic, inject_synthetic

save_synthetic(X_generated, "wgangp_seed42")        # -> data/synthetic/wgangp_seed42.npy
generated = load_synthetic("wgangp_seed42", n_features=29)
augmented = inject_synthetic(split.X_train, split.y_train, generated)
```

`load_synthetic` valide la forme ; `inject_synthetic` **rejette les NaN/inf** (générateur
divergé) et ne reçoit aucun jeu de test — elle est structurellement incapable de le
contaminer. `augmented.is_synthetic` garde l'origine de chaque ligne auditable après mélange.

---

## 5. Ce que le ML Engineer fournit en retour (spécifications côté modèle)

### 5.1 Pour le dataset image

- **Shape attendue (gabarit)** : `(batch, 1, 28, 28)` Fashion-MNIST / `(batch, 3, 32, 32)` CIFAR-10
  → **livré en `(batch, 1, 32, 32)` et `(batch, 3, 32, 32)`** — cf. conflit **C2**
- **Range de normalisation** : `[-1, 1]` — ✅ **conforme**
- **Volume minimum pour 5 seeds × N époques** : *(à confirmer par le ML Engineer)* — le DE
  fournit 60 000 / 50 000 images d'entraînement, intégralement rejouées à chaque seed.

### 5.2 Pour le dataset tabulaire

- **Shape attendue** : `(batch, n_features)` → **`n_features = 29`**
- **Range de normalisation** : gabarit mentionne « z-score, ou `[-1, 1]` si architecture à
  sortie `Tanh()` » → **livré en z-score, valeurs observées `[-82,4 ; +103,5]`**
  → ⚠️ **conflit C1, bloquant**
- **Type de labels** : binaire 0/1, sans valeurs manquantes — ✅ **conforme**
  (array `y` séparé, la colonne source s'appelle `Class`)
- **Volume minimum de la classe minoritaire** : **394 fraudes réelles** disponibles pour
  entraîner le GAN. *(Le ML Engineer doit confirmer si ce volume lui suffit — voir Q1 en §6.)*
- **Split réservé à l'évaluation aval** : **test de 56 962 lignes dont 98 fraudes,
  100 % réel**, garanti sans aucun échantillon synthétique — ✅ **conforme**, sous test
  automatisé (`tests/test_leakage.py`).

---

## 6. Points de vigilance / questions ouvertes

### 🔴 C1 — `Tanh()` en sortie de générateur est incompatible avec le tabulaire *(bloquant)*

Les features tabulaires sont **standardisées z-score**, donc **non bornées** : observé
`[-82,4 ; +103,5]` sur le train. Une sortie `Tanh()` ne peut produire que `[-1, 1]`.

Le générateur serait alors **structurellement incapable** d'atteindre plus de 99 % de
l'espace des fraudes réelles. Rien ne planterait : l'entraînement convergerait vers des
échantillons tous tassés au centre de la distribution, et le classifieur aval verrait ses
performances stagner sans cause visible.

**Trois issues possibles** :

| Option | Effet | Avis DE |
|---|---|---|
| **A.** Générateur tabulaire à **sortie linéaire** (pas de `Tanh`) | Aucun changement de données | ✅ **Recommandé.** `Tanh` est une convention *image*, liée au fait que les pixels sont naturellement bornés. Rien ne l'impose sur du tabulaire. |
| **B.** Scaler `MinMax` vers `[-1, 1]` côté DE | Permet de garder `Tanh` | ⚠️ Écrase les queues : les fraudes extrêmes — les plus informatives — se tasseraient près des bornes. |
| **C.** Conserver z-score + clipper les sorties | Simple | ❌ Introduit un biais artificiel aux bornes. |

**→ Décision attendue du ML Engineer.** L'option A ne demande aucun travail côté DE ;
l'option B est un changement d'une ligne dans `config.py`, mais dégrade le signal.

### 🟠 C2 — Résolution Fashion-MNIST : 28×28 ou 32×32 ?

Le gabarit (§5.1) annonce `(batch, 1, 28, 28)` ; je livre `(batch, 1, 32, 32)`.

**Avis DE : garder 32×32.** 32 est une puissance de 2, ce qui donne une chaîne
d'*upsampling* propre `4 → 8 → 16 → 32` en convolutions transposées stride-2. Avec 28,
l'arithmétique kernel/stride/padding devient irrégulière et l'architecture ne peut plus être
partagée avec CIFAR-10 — ce qui affaiblirait la comparaison entre les deux datasets.

**Coût du changement : nul dans les deux sens.** Vérifié :
`get_image_dataloader("fashion_mnist", image_size=28)` → `(8, 1, 28, 28)`, plage `[-1, 1]`.

**→ Décision attendue du ML Engineer.**

### 🟡 C3 — `float64` côté tabulaire, `float32` attendu par PyTorch

Les arrays `.npy` tabulaires sont en `float64` (sortie naturelle de scikit-learn) ; les
tenseurs images sont déjà en `float32`. `torch.tensor(X_train)` produirait donc du `float64`
et déclencherait une erreur de dtype au premier passage dans le modèle.

**Contournement immédiat côté ML** : `X_train.astype(np.float32)`.
**Ou côté DE** : conversion à la sauvegarde, ~2 lignes. Divise aussi par deux la taille disque.

**→ Préférence à indiquer par le ML Engineer.**

### 🟡 C4 — Pas de split de validation

Livré : **train / test** uniquement, sur les deux pipelines. Le gabarit (§2.2, §3.5) laisse
la question ouverte.

Pour un GAN, une validation n'est pas indispensable — il n'y a pas d'*early stopping* sur une
métrique de validation au sens classique. Elle le devient si le ML Engineer veut **sélectionner
un checkpoint** ou **régler les hyperparamètres du classifieur aval** sans toucher au test.

**Coût côté DE** : faible, `splits.py` gère déjà des splits stratifiés déterministes.
**→ Dire si un split de validation est nécessaire, et sur quel pipeline.**

### 🟡 Q1 — 394 fraudes suffisent-elles à entraîner le GAN tabulaire ?

C'est le volume réel disponible, non négociable : les 98 autres fraudes sont dans le test et
doivent y rester. Apprendre une distribution à 29 dimensions depuis 394 exemples est le point
dur du livrable secondaire, et probablement là où WGAN-GP se distinguera de DCGAN.

**→ Le ML Engineer confirme la faisabilité, ou propose un plan B** (validation croisée sur le
GAN, augmentation classique en complément, etc.).

### 🟡 Q2 — Version de scikit-learn pour lire le scaler

Le `StandardScaler` est sérialisé par joblib. Un chargement avec une version différente de
celle qui l'a produit émet un `InconsistentVersionWarning` — **constaté en pratique** entre
sklearn 1.9.0 (écriture) et 1.8.0 (lecture).

`requirements.txt` déclare `scikit-learn>=1.3`, sans borne haute. **Recommandation : figer une
version commune** aux deux rôles, sinon `inverse_transform` peut se comporter différemment
d'un poste à l'autre — silencieusement.

**→ Se mettre d'accord sur une version et l'épingler.**

---

## 7. Historique des révisions

| Date | Modifié par | Changement |
|---|---|---|
| — | ML Engineer | Version initiale du contrat (gabarit) |
| 2026-08-19 | Data Engineer | Remplissage §1–§4 avec les valeurs livrées et vérifiées ; annotation §5 ; ouverture des conflits C1–C4 et questions Q1–Q2 en §6 |
