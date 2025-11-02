from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)

    # Créer le dossier model si n’existe pas
    os.makedirs("model", exist_ok=True)
    joblib.dump(clf, "model/iris_model.pkl")
    print("Modèle entraîné et sauvegardé !")

if __name__ == "__main__":
    train()