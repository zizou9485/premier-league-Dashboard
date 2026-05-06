import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Premier League Dashboard", layout="wide")

# On charge et entraine le modéles
@st.cache_resource
def charger_modele():
    df = pd.read_csv("PL_10_ans.csv")
    df_ml = df[["HomeTeam", "AwayTeam", "FTR"]].dropna()

    enc_equipes = LabelEncoder()
    enc_equipes.fit(pd.concat([df_ml["HomeTeam"], df_ml["AwayTeam"]]))
    df_ml["HomeTeam_ID"] = enc_equipes.transform(df_ml["HomeTeam"])
    df_ml["AwayTeam_ID"] = enc_equipes.transform(df_ml["AwayTeam"])

    enc_resultat = LabelEncoder()
    df_ml["Resultat_ID"] = enc_resultat.fit_transform(df_ml["FTR"])

    X = df_ml[["HomeTeam_ID", "AwayTeam_ID"]]
    y = df_ml["Resultat_ID"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    modele = RandomForestClassifier(n_estimators=100, random_state=42)
    modele.fit(X_train, y_train)

    equipes = sorted(pd.concat([df_ml["HomeTeam"], df_ml["AwayTeam"]]).unique())
    return modele, enc_equipes, enc_resultat, equipes


@st.cache_data
def charger_stats():
    return pd.read_csv("classement_final_complet.csv")


modele, enc_equipes, enc_resultat, equipes = charger_modele()
stats = charger_stats()


st.sidebar.title("Premier League")
page = st.sidebar.radio("Navigation", ["Prédiction", "Statistiques"])

# la page de prédiction
if page == "Prédiction":
    st.title("Prédiction de match")
    st.markdown("Sélectionne deux équipes et le modèle prédit le résultat à partir de 10 ans de données.")

    col1, col2 = st.columns(2)
    with col1:
        domicile = st.selectbox("Équipe domicile", equipes, index=equipes.index("Arsenal"))
    with col2:
        exterieur = st.selectbox("Équipe extérieur", equipes, index=equipes.index("Liverpool"))

    if domicile == exterieur:
        st.warning("Sélectionne deux équipes différentes.")
    else:
        if st.button("Prédire"):
            id_dom = enc_equipes.transform([domicile])[0]
            id_ext = enc_equipes.transform([exterieur])[0]
            pred_id = modele.predict([[id_dom, id_ext]])[0]
            resultat = enc_resultat.inverse_transform([pred_id])[0]
            proba = modele.predict_proba([[id_dom, id_ext]])[0]
            idx = {label: i for i, label in enumerate(enc_resultat.classes_)}

            if resultat == "H":
                gagnant = domicile
                confiance = proba[idx["H"]]
                couleur = "normal"
            elif resultat == "A":
                gagnant = exterieur
                confiance = proba[idx["A"]]
                couleur = "normal"
            else:
                gagnant = "Match Nul"
                confiance = proba[idx["D"]]
                couleur = "normal"

            st.markdown("---")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Résultat prédit", gagnant)
            col_b.metric("Confiance", f"{confiance * 100:.1f}%")
            col_c.metric("Matchs analysés", "3 800")

            st.markdown("---")
            st.subheader("Probabilités détaillées")
            proba_df = pd.DataFrame({
                "Issue": [domicile, "Match Nul", exterieur],
                "Probabilité (%)": [
                    round(proba[idx["H"]] * 100, 1),
                    round(proba[idx["D"]] * 100, 1),
                    round(proba[idx["A"]] * 100, 1),
                ]
            })
            st.bar_chart(proba_df.set_index("Issue"))

# La page des statistiques
elif page == "Statistiques":
    st.title("Statistiques — Saison 2021-2022")

    st.subheader("Bilan attaque / défense")
    equipe_choisie = st.selectbox("Filtrer une équipe", ["Toutes"] + stats["Domicile"].tolist())

    if equipe_choisie != "Toutes":
        df_filtre = stats[stats["Domicile"] == equipe_choisie]
    else:
        df_filtre = stats

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(df_filtre))
    ax.barh(df_filtre["Domicile"], df_filtre["Buts_Marqués"], color="#2ecc71", label="Buts marqués")
    ax.barh(df_filtre["Domicile"], df_filtre["Buts_Encaissés"], color="#e74c3c", alpha=0.7, label="Buts encaissés")
    ax.set_xlabel("Nombre de buts")
    ax.legend()
    ax.invert_yaxis()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Tableau complet")
    stats_affichage = stats.copy()
    stats_affichage["Différence"] = stats_affichage["Buts_Marqués"] - stats_affichage["Buts_Encaissés"]
    st.dataframe(stats_affichage, use_container_width=True)