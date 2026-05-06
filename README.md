# Premier League Dashboard

Dashboard interactif pour explorer les données de la Premier League et prédire les résultats de matchs. Fait suite au projet d'analyse et au modèle de prédiction Random Forest entraîné sur 10 ans de données.

## Ce que ça fait

Deux pages accessibles depuis la sidebar :

- **Prédiction** : on choisit deux équipes, le modèle prédit le résultat et affiche les probabilités pour chaque issue (victoire domicile, nul, victoire extérieur)
- **Statistiques** : bilan attaque / défense de la saison 2021-2022 avec filtrage par équipe et tableau complet

## Installation

```bash
git clone https://github.com/zizou9485/premier-league-dashboard.git
cd premier-league-dashboard
pip install -r requirements.txt
```

## Utilisation

```bash
python -m streamlit run app.py
```

## Fichiers

```
├── app.py                        # Application Streamlit
├── PL_10_ans.csv                 # Données 2013-2023 (entraînement du modèle)
├── classement_final_complet.csv  # Stats saison 2021-2022
├── requirements.txt
└── .gitignore
```

## Technologies

- Python 3.12
- Streamlit
- Scikit-learn (Random Forest)
- Pandas / Matplotlib
