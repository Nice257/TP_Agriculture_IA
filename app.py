
import joblib
import pandas as pd
import streamlit as st

VERT_FONCE = "#2e7d32"
VERT_MOYEN = "#4caf50"
VERT_CLAIR = "#a5d6a7"
FOND_VERT = "#e8f5e9"
BLANC = "#ffffff"

st.set_page_config(page_title="Prédiction des Récoltes au Burundi", page_icon="🌾", layout="centered")

st.markdown(
    "<style>\n"
    ".stApp { background-color: #ffffff; color: #2e7d32; }\n"
    "h1, h2, h3, label, .stMarkdown { color: #2e7d32 !important; }\n"
    "div.stButton > button { background-color: #2e7d32; color: white; border-radius: 8px; border: 1px solid #2e7d32; font-weight: 700; }\n"
    "div.stButton > button:hover { background-color: #4caf50; color: white; border: 1px solid #4caf50; }\n"
    ".resultat-bon { background-color: #e8f5e9; border-left: 8px solid #2e7d32; padding: 20px; border-radius: 8px; color: #2e7d32; font-size: 28px; font-weight: 800; text-align: center; }\n"
    ".resultat-mauvais { background-color: #ffebee; border-left: 8px solid #c62828; padding: 20px; border-radius: 8px; color: #c62828; font-size: 28px; font-weight: 800; text-align: center; }\n"
    "[data-testid='stMetricValue'] { color: #2e7d32; }\n"
    "</style>",
    unsafe_allow_html=True
)

@st.cache_resource
def charger_objets():
    modeles = {
        "Arbre de Décision": joblib.load("model_arbre.pkl"),
        "Forêt Aléatoire": joblib.load("model_foret.pkl"),
        "Régression Logistique": joblib.load("model_regression.pkl")
    }
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("encoders.pkl")
    return modeles, scaler, encoders

modeles, scaler, encoders = charger_objets()

METRIQUES_MODELES = {
    "Arbre de Décision": {"Accuracy": 0.89873418, "F1": 0.94557823, "AUC": 0.75197740},
    "Forêt Aléatoire": {"Accuracy": 0.93354430, "F1": 0.96540362, "AUC": 0.72251816},
    "Régression Logistique": {"Accuracy": 0.93354430, "F1": 0.96551724, "AUC": 0.85536723}
}

colonnes_modeles = ["annee", "saison", "province", "culture", "altitude_m", "pluviometrie_mm",
                    "temperature_moy_C", "superficie_ha", "utilisation_engrais", "acces_irrigation", "nb_menages"]
colonnes_continues = ["annee", "altitude_m", "pluviometrie_mm", "temperature_moy_C", "superficie_ha", "nb_menages"]

st.markdown("<h1 style='color:#2e7d32;'>🌾 Prédiction des Récoltes au Burundi</h1>", unsafe_allow_html=True)


modele_choisi = st.selectbox("Choisir le modèle", list(modeles.keys()))

col1, col2 = st.columns(2)
with col1:
    province = st.selectbox("Province", list(encoders["province"].classes_))
    culture = st.selectbox("Culture", list(encoders["culture"].classes_))
    saison = st.selectbox("Saison", list(encoders["saison"].classes_))
    altitude_m = st.number_input("Altitude (m)", min_value=0, max_value=3000, value=1500, step=10)
    pluviometrie_mm = st.number_input("Pluviométrie (mm)", min_value=0.0, max_value=2000.0, value=850.0, step=10.0)
with col2:
    temperature_moy_C = st.number_input("Température moyenne (°C)", min_value=0.0, max_value=40.0, value=21.0, step=0.1)
    superficie_ha = st.number_input("Superficie cultivée (ha)", min_value=0.1, max_value=100.0, value=2.0, step=0.1)
    utilisation_engrais_texte = st.selectbox("Engrais", ["Oui", "Non"])
    acces_irrigation_texte = st.selectbox("Irrigation", ["Oui", "Non"])
    nb_menages = st.number_input("Nombre de ménages", min_value=1, max_value=10000, value=100, step=1)

if st.button("🔍 Prédire la récolte"):
    nouvelle_donnee = pd.DataFrame({
        "annee": [2024], "saison": [saison], "province": [province], "culture": [culture],
        "altitude_m": [altitude_m], "pluviometrie_mm": [pluviometrie_mm],
        "temperature_moy_C": [temperature_moy_C], "superficie_ha": [superficie_ha],
        "utilisation_engrais": [1 if utilisation_engrais_texte == "Oui" else 0],
        "acces_irrigation": [1 if acces_irrigation_texte == "Oui" else 0],
        "nb_menages": [nb_menages]
    })
    for colonne in ["saison", "province", "culture"]:
        nouvelle_donnee[colonne] = encoders[colonne].transform(nouvelle_donnee[colonne])
    nouvelle_donnee = nouvelle_donnee[colonnes_modeles]
    nouvelle_donnee[colonnes_continues] = scaler.transform(nouvelle_donnee[colonnes_continues])

    modele = modeles[modele_choisi]
    prediction = int(modele.predict(nouvelle_donnee)[0])
    probabilites = modele.predict_proba(nouvelle_donnee)[0]
    probabilite = probabilites[prediction] * 100
    if prediction == 1:
        st.markdown(f"<div class='resultat-bon'>✅ BONNE RÉCOLTE ({probabilite:.0f}%)</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='resultat-mauvais'>❌ MAUVAISE RÉCOLTE ({probabilite:.0f}%)</div>", unsafe_allow_html=True)

    st.subheader("Métriques du modèle sélectionné")
    m1, m2, m3 = st.columns(3)
    metriques = METRIQUES_MODELES[modele_choisi]
    m1.metric("Accuracy", f"{metriques['Accuracy']:.3f}")
    m2.metric("F1-score", f"{metriques['F1']:.3f}")
    m3.metric("AUC", f"{metriques['AUC']:.3f}")
