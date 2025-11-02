from sklearn.datasets import load_wine # <-- CHANGEMENT ICI
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import argparse

# Définition du chemin du modèle existant
MODEL_PATH = "model/best_model.pkl" # Nom conservé pour la compatibilité démo

def train(n_estimators=10):
    # 1. Préparation des données (Jeu de données Wine)
    data = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

    # 2. Entraînement du modèle
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    clf.fit(X_train, y_train)

    # 3. Évaluation du modèle
    y_pred = clf.predict(X_test)
    current_score = accuracy_score(y_test, y_pred)
    print(f"Score de précision (n_estimators={n_estimators}): {current_score:.4f}")

    # 4. Chargement et évaluation du modèle existant pour comparaison
    best_score = -1.0
    if os.path.exists(MODEL_PATH):
        try:
            old_clf = joblib.load(MODEL_PATH)
            old_y_pred = old_clf.predict(X_test)
            best_score = accuracy_score(y_test, old_y_pred)
            print(f"Score du modèle précédent: {best_score:.4f}")
        except Exception:
            print("Impossible d'évaluer l'ancien modèle (probablement le premier entraînement).")
            best_score = -1.0
        
    # 5. Mise à jour conditionnelle
    if current_score > best_score:
        os.makedirs("model", exist_ok=True)
        joblib.dump(clf, MODEL_PATH)
        print("✅ Nouveau modèle sauvegardé (score amélioré) !")
    else:
        print("❌ Score non amélioré. Le modèle précédent est conservé.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîner le modèle Vin.")
    parser.add_argument("--n_estimators", type=int, default=10,
                        help="Nombre d'arbres dans la forêt aléatoire.")
    args = parser.parse_args()
    
    train(n_estimators=args.n_estimators)