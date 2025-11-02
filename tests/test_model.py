import joblib
import numpy as np
import os
from sklearn.datasets import load_wine # Ajout de load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Ce test vérifie que le fichier modèle existe
def test_model_exists():
    assert os.path.exists("model/best_model.pkl"), "Le modèle n'existe pas !"

# Ce test vérifie que le modèle charge correctement et accepte 13 caractéristiques en entrée
def test_prediction_shape():
    model = joblib.load("model/best_model.pkl")
    # Vecteur de 13 caractéristiques pour le jeu de données Wine
    X = np.array([[13.0, 2.0, 2.5, 20.0, 100.0, 2.5, 2.5, 0.3, 1.5, 5.0, 1.0, 3.0, 800.0]])
    pred = model.predict(X)
    assert pred.shape == (1,), "La prédiction doit retourner un vecteur de taille 1 (13 entrées requises)"

# NOUVEAU TEST : Vérifie que le modèle sauvegardé a une précision minimale
def test_model_minimum_accuracy():
    # Recharger les données de test (DOIT UTILISER LE MÊME RANDOM_STATE QUE train_model.py)
    data = load_wine()
    _, X_test, _, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

    # Charger le modèle
    model = joblib.load("model/best_model.pkl")

    # Calculer la précision
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    MIN_ACCURACY = 0.90 # Seuil de qualité que nous avons défini
    
    # Assert
    assert accuracy >= MIN_ACCURACY, f"La précision du modèle est trop faible ({accuracy:.4f}), attendu >= {MIN_ACCURACY}"