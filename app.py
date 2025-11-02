import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Prédiction Vin - Démo CI/CD", layout="centered")

# --- Styles ---
st.markdown("""
<style>
    .stSlider > label {
        font-weight: bold;
        color: #B0282E; /* Couleur Bordeaux/Vin */
    }
    .stButton>button {
        background-color: #B0282E;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 1.2rem;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #8C2025;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)
# --- Fin Styles ---

st.title("🍷 Prédiction de Type de Vin - Démo CI/CD")
st.markdown("Ajustez les paramètres chimiques du vin pour prédire sa classe (0, 1 ou 2).")

# Le modèle doit toujours être nommé best_model.pkl pour la démo, même s'il s'agit du modèle Wine
try:
    model = joblib.load("model/best_model.pkl")
    st.success("Modèle de classification des vins chargé avec succès.")
except FileNotFoundError:
    st.error("Erreur : Le fichier 'model/best_model.pkl' est introuvable. Veuillez l'entraîner d'abord (python model/train_model.py).")
    st.stop()


# Définition des 13 caractéristiques du jeu de données Wine
st.header("Paramètres Chimiques")

col1, col2, col3 = st.columns(3)

with col1:
    alcohol = st.slider("1. Alcool", 11.0, 14.5, 13.0, 0.1)
    malic_acid = st.slider("2. Acide Malique", 0.5, 6.0, 2.5, 0.1)
    ash = st.slider("3. Cendres", 1.5, 3.5, 2.4, 0.1)
    alcalinity_of_ash = st.slider("4. Alcalinité des Cendres", 10.0, 30.0, 18.0, 0.5)
    magnesium = st.slider("5. Magnesium", 70, 160, 100, 5)

with col2:
    total_phenols = st.slider("6. Total Phénols", 0.5, 4.0, 2.5, 0.1)
    flavanoids = st.slider("7. Flavanoides", 0.0, 5.0, 2.5, 0.1)
    nonflavanoid_phenols = st.slider("8. Phénols Non-flavanoïdes", 0.1, 0.6, 0.3, 0.01)
    proanthocyanins = st.slider("9. Proanthocyanines", 0.5, 4.0, 1.5, 0.1)

with col3:
    color_intensity = st.slider("10. Intensité de la Couleur", 1.0, 13.0, 5.0, 0.5)
    hue = st.slider("11. Teinte (Hue)", 0.5, 1.8, 1.0, 0.01)
    od280_od315_of_diluted_wines = st.slider("12. OD280/OD315", 1.0, 4.5, 3.0, 0.1)
    proline = st.slider("13. Proline", 200, 1700, 800, 10)


if st.button("Prédire la Classe de Vin"):
    # Création du vecteur de 13 caractéristiques
    X = np.array([[
        alcohol, malic_acid, ash, alcalinity_of_ash, magnesium,
        total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins,
        color_intensity, hue, od280_od315_of_diluted_wines, proline
    ]])

    try:
        pred = model.predict(X)
        st.success(f"🍾 Classe de Vin Prédite : Classe {pred[0]}")
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")


# import streamlit as st
# import joblib
# import numpy as np

# st.title("Prédiction Iris - Démo CI/CD")

# model = joblib.load("model/iris_model.pkl")

# sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.0)
# sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.0)
# petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0)
# petal_width = st.slider("Petal Width", 0.1, 2.5, 1.0)

# if st.button("Prédire"):
#     X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
#     pred = model.predict(X)
#     st.success(f"Classe prédite : {pred[0]}")