# Détection des fraudes bancaires par analyse massive des données et Intelligence Artificielle

Projet de fin d'études — **Master Intelligence Artificielle et Cybersécurité** (promotion 2024–2026)
Faculté des Sciences de Kénitra

- **Réalisé par** : Wafae Bouajaja
- **Encadré par** : Prof. Hajar Filali

## Description

Les fraudes par carte bancaire sont extrêmement rares (0,17 % des transactions dans le dataset réel), ce qui rend leur détection difficile : un modèle qui prédit toujours « normal » atteint déjà 99,8 % d'accuracy sans détecter aucune fraude.

Ce projet compare **5 algorithmes** sur deux configurations de données (un dataset réel très déséquilibré et un dataset équilibré) afin d'identifier le meilleur compromis entre détection des fraudes et fausses alertes :

| Type | Algorithmes |
|---|---|
| Supervisé | Random Forest, SVM |
| Deep Learning | RNN, LSTM (testé uniquement sur le dataset déséquilibré) |
| Non supervisé | K-Means |

## Données

| Dataset | Fichier | Transactions | Fraudes | Variables |
|---|---|---|---|---|
| Déséquilibré | `creditcard.csv` | 284 807 | 492 (0,17 %) | Time, V1–V28, Amount, Class |
| Équilibré | `creditcard_2023.csv` | 568 630 | 284 315 (50 %) | id, V1–V28, Amount, Class |

Les variables V1 à V28 sont anonymisées (issues d'une ACP). `Class` vaut 0 pour une transaction normale et 1 pour une fraude.

> Les fichiers CSV dépassent la limite de 100 Mo de GitHub : ils **ne sont pas inclus** dans ce dépôt.
> Téléchargez-les sur Kaggle et placez-les dans le dossier `data/` :
> - Dataset déséquilibré : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
> - Dataset équilibré (2023) : https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023

## Méthodologie

Pipeline : chargement des données → prétraitement et normalisation → gestion du déséquilibre → classification → évaluation et décision.

- **Normalisation** : `StandardScaler`
- **Dataset équilibré** : split 80/20 stratifié, aucun rééchantillonnage nécessaire
- **Dataset déséquilibré** : traitement adapté à chaque algorithme
  - Random Forest : `class_weight='balanced'`
  - SVM : SMOTE + sous-échantillonnage
  - RNN / LSTM / K-Means : tirage aléatoire équilibré (fraudes conservées intégralement)
- **Seuil de décision** : sept seuils testés (0,1 à 0,7) pour maximiser le F1-Score
- **Métriques** : Accuracy, Précision, Recall, F1-Score, FPR, matrice de confusion, Silhouette Score (K-Means)

## Résultats

### Dataset déséquilibré (`creditcard.csv`)

| Modèle | Test set | Précision | Recall | F1-Score |
|---|---|---|---|---|
| Random Forest | 56 962 (réel complet) | 92,2 % | 84,7 % | 88,3 % |
| SVM | 56 962 (réel complet) | 10,7 % | 86,7 % | 19,1 % |
| K-Means | 197 (équilibré) | 100 % | 35,7 % | 52,6 % |
| RNN | 197 (équilibré) | 94,7 % | 91,8 % | 93,3 % |
| LSTM | 499 (équilibré) | 93,7 % | 89,9 % | 91,8 % |

### Dataset équilibré (`creditcard_2023.csv`)

| Modèle | Précision | Recall | F1-Score |
|---|---|---|---|
| Random Forest | 99,9 % | 100 % | 99,9 % |
| RNN | 99,8 % | 100 % | 99,9 % |
| SVM linéaire | 96,9 % | 95,8 % | 96,3 % |
| K-Means | 94,5 % | 86,0 % | 90,1 % |

### Conclusions principales

- **Random Forest** : choix le plus fiable en conditions réelles déséquilibrées (7 fausses alertes et 15 fraudes manquées sur 56 962 transactions).
- **SVM** : le plus sensible au déséquilibre malgré SMOTE (706 fausses alertes).
- **K-Means** : limites confirmées pour un problème supervisé par nature.
- **RNN / LSTM** : pistes prometteuses, mais évaluées sur des sous-ensembles équilibrés, à valider sur les données réelles complètes.

### Comparaison avec les travaux connexes (2025)

| Référence | Algorithme | F1-Score |
|---|---|---|
| Popova & Gardi (2025) | Random Forest | 85,8 % |
| Zhou et al. (2025) | RNN-LSTM hybride | 85,3 % |
| **Ce projet** | **Random Forest** | **88,3 %** |

## Structure du dépôt

```
├── notebooks/
│   ├── Code_creditcard.ipynb        # dataset déséquilibré (RF, SVM, RNN, LSTM, K-Means)
│   └── Code_creditcard2023.ipynb    # dataset équilibré (RF, SVM, RNN, K-Means)
├── docs/
│   ├── rapport_PFE.pdf              # mémoire complet
│   └── presentation_PFE.pptx        # support de soutenance
├── data/                            # à remplir avec les CSV (non versionné)
├── requirements.txt
└── README.md
```

## Installation et utilisation

1. Cloner le dépôt :
```bash
   git clone https://github.com/WafaeBouajaja/detection-fraude-carte-bancaire.git
   cd detection-fraude-carte-bancaire
```
2. Installer les dépendances :
```bash
   pip install -r requirements.txt
```
3. Télécharger les CSV (voir section *Données*) et les placer dans `data/`.
4. Dans chaque notebook, adapter le chemin de lecture du fichier. Les notebooks ont été exécutés sur Kaggle, remplacez donc `/kaggle/input/...` par :
```python
   df = pd.read_csv('../data/creditcard.csv')        # ou ../data/creditcard_2023.csv
```
5. Lancer Jupyter :
```bash
   jupyter notebook
```

## Documents

- [Rapport complet (PDF)](docs/rapport_PFE.pdf)
- [Présentation de soutenance](docs/presentation_PFE.pptx)

## Technologies

Python · pandas · NumPy · scikit-learn · imbalanced-learn (SMOTE) · TensorFlow / Keras
