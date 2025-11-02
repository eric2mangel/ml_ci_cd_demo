🍷 Démonstration de Pipeline CI/CD pour le Machine Learning

Ce projet est une démonstration simple d'un workflow d'Intégration et de Déploiement Continus (CI/CD) appliqué à un modèle de Machine Learning. Il utilise le jeu de données Wine de Scikit-learn et un modèle Random Forest pour illustrer comment les tests unitaires et la validation des scores peuvent être automatisés via GitHub Actions.

🎯 Objectifs de la Démo

Automatisation de la Qualité : Vérifier que les tests unitaires et la précision minimale du modèle sont respectés à chaque push.

Entraînement Conditionnel : Mettre à jour le modèle sauvegardé (iris_model.pkl) uniquement si un nouvel entraînement (avec de nouveaux hyperparamètres) donne un score de précision supérieur.

Déploiement Simplifié : Mettre en place un pipeline prêt au déploiement continu.

📁 Structure du Projet

ml_ci_cd_demo/
│
├─ app.py                  # Interface utilisateur Streamlit pour les prédictions (13 features Wine)
├─ model/
│   ├─ train_model.py      # Script d'entraînement, d'évaluation et de sauvegarde conditionnelle
│   └─ iris_model.pkl      # Le modèle sauvegardé (mis à jour uniquement si le score s'améliore)
├─ tests/
│   └─ test_model.py       # Tests unitaires (existence du fichier, forme d'entrée, précision minimale > 0.90)
├─ requirements.txt        # Dépendances Python (scikit-learn, streamlit, pytest)
├─ README.md               # Ce fichier
└─ .github/
    └─ workflows/
        └─ ci_cd.yml       # Pipeline GitHub Actions (Installe, Entraîne, Teste)


🛠️ Configuration et Lancement Local

Prérequis

Python 3.8+

Git

1. Installation

Créez et activez votre environnement virtuel, puis installez les dépendances :

# Créer l'environnement virtuel
python -m venv venv

# Activer l'environnement (Linux/Mac)
source venv/bin/activate

# Activer l'environnement (Windows/PowerShell)
.\venv\Scripts\Activate

# Installer les dépendances
pip install -r requirements.txt


2. Entraînement Initial du Modèle

Lancez l'entraînement pour créer la première version du modèle iris_model.pkl. Nous utilisons 100 estimateurs pour une meilleure précision.

python model/train_model.py --n_estimators 100


Note : Le modèle sera sauvegardé si son score est meilleur que le score du modèle existant.

3. Exécution des Tests Unitaires

Exécutez pytest pour valider que le modèle sauvegardé respecte les standards de qualité (existence, format et précision minimale de 0.90).

pytest tests/


4. Lancement de l'Application (Facultatif)

Testez l'interface utilisateur locale pour vérifier que le modèle chargé peut effectuer des prédictions :

streamlit run app.py


🌐 Pipeline CI/CD GitHub Actions

Le fichier .github/workflows/ci_cd.yml définit la logique du pipeline :

Déclenchement : Le pipeline se lance à chaque push sur la branche main ou à chaque Pull Request.

Entraînement et Sauvegarde : Le script train_model.py est lancé pour générer ou mettre à jour conditionnellement le modèle.

Validation : L'étape Run tests exécute pytest tests/.

Succès : Si les tests passent, le pipeline est vert et le déploiement est simulé (ou pourrait se poursuivre vers un service comme Hugging Face Spaces).

Pour déclencher le pipeline, committez et poussez vos modifications vers votre dépôt GitHub :

git add .
git commit -m "chore: configuration finale de la démo wine"
git push origin main
