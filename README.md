Ce projet implémente un modèle de détection de fraude sur les transactions par carte de crédit à l'aide de LightGBM et XGBoost ( défi : Manque énorme de données pour la classe positive)

Structure du Projet

    main.py : Lance l'ensemble du pipeline (prétraitement, entraînement et évaluation).
    data_processing_n_analysis.py : Prépare les données et contient les fonctions d'analyse.
    Train.py : Entraîne les modèles LightGBM et XGBoost.
    Test.py : Évalue les performances des modèles.
    references.txt : Contient les informations sur les données utilisées.


Installez les dépendances avec :
pip install -r requirements.txt

Utilisation :
python main.py

Résultats :
    ROC AUC : 98%
    Précision : 91%
    Rappel (classe positive) : 85%
