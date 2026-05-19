#  Prédiction des Récoltes au Burundi

> Projet de Travaux Pratiques — Intelligence Artificielle Appliquée à l'Agriculture  
> BAC 4 Génie Logiciel — Université Polytechnique de Gitega (UPG)

---

## Description

Ce projet utilise des algorithmes de **Machine Learning** pour prédire si une récolte agricole sera **bonne ou mauvaise** au Burundi, en fonction de facteurs mesurables comme la pluviométrie, la température, l'altitude, l'utilisation d'engrais et l'accès à l'irrigation.

L'application web permet à n'importe quel utilisateur de saisir les caractéristiques d'une parcelle agricole et d'obtenir en temps réel une prédiction avec sa probabilité.

---

##  Objectifs

- Analyser et explorer les données agricoles de 15 provinces du Burundi (2015–2023)
- Construire et comparer 3 modèles de Machine Learning
- Déployer une application web interactive pour la prédiction

---

##  Dataset

| Caractéristique | Détail |
|---|---|
| Fichier | `agriculture_burundi.csv` |
| Lignes | 1 620 observations |
| Colonnes | 14 variables |
| Période | 2015 à 2023 |
| Provinces | 15 provinces du Burundi |
| Cultures | Maïs, Haricot, Manioc, Patate douce, Sorgho, Bananier |
| Saisons | A (mars–juin) et B (septembre–décembre) |
| Variable cible | `bonne_recolte` (1 = bonne, 0 = mauvaise) |

---

##  Modèles entraînés

| Modèle | Description |
|---|---|
|  Arbre de Décision | Modèle interprétable avec visualisation des règles |
|  Forêt Aléatoire | Ensemble de 100 arbres — meilleure robustesse |
| Régression Logistique | Modèle linéaire avec interprétation des coefficients |

---

##  Structure du projet

```
TP_Agriculture_IA/
│
├── TP_Agriculture.ipynb      # Notebook Jupyter complet (40 cellules)
├── app.py                    # Application web Streamlit
├── agriculture_burundi.csv   # Dataset agricole
│
├── model_arbre.pkl           # Modèle Arbre de Décision sauvegardé
├── model_foret.pkl           # Modèle Forêt Aléatoire sauvegardé
├── model_regression.pkl      # Modèle Régression Logistique sauvegardé
├── scaler.pkl                # StandardScaler sauvegardé
├── encoders.pkl              # LabelEncoders sauvegardés
│
└── README.md                 # Ce fichier
```

---

##  Installation et lancement

### 1. Cloner le dépôt

```bash
git clone https://github.com/Nice257/TP_Agriculture_IA.git```

### 2. Installer les dépendances

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit
```

### 3. Lancer l'application web

```bash
streamlit run app.py
```

L'application s'ouvre automatiquement dans votre navigateur à l'adresse :
```
http://localhost:8501
```

### 4. Exécuter le notebook (optionnel)

Ouvrez `TP_Agriculture.ipynb` dans VS Code ou Jupyter et faites **"Restart Kernel and Run All"**.

---

##  Application déployée

 **URL de l'application** : PAS ENCORE

---

##  Aperçu de l'application

![Aperçu de l'application](images/capture.png)

---

## Technologies utilisées

- **Python 3**
- **pandas** — manipulation des données
- **numpy** — calculs numériques
- **matplotlib / seaborn** — visualisations
- **scikit-learn** — Machine Learning
- **joblib** — sauvegarde des modèles
- **Streamlit** — application web
- **VS Code + Jupyter** — environnement de développement

---

##  Auteur

- **Nom** : NSABIYUMVA NICE STELLA
- **Promotion** : BAC 4 Génie Logiciel
- **Université** : Université Polytechnique de Gitega (UPG)
- **Année académique** : 2025–2026