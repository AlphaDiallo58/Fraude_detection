from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt

def test_model(model, X_test, y_test, threshold=0.5):
    """
    Teste le modèle sur les données de test et renvoie les résultats d'évaluation.

    """
    
    if isinstance(model, lgb.Booster):
        y_pred_proba = model.predict(X_test)
    elif isinstance(model, xgb.XGBClassifier):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        raise ValueError("Le modèle n'est ni un modèle LightGBM ni XGBoost")
        
    y_pred = (y_pred_proba >= threshold).astype(int)

    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"AUC-ROC : {roc_auc:.4f}")

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc_value = auc(fpr, tpr)

    # Graphique ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc_value:.4f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonale de la courbe aléatoire
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Courbe ROC')
    plt.legend(loc='lower right')
    plt.show()
    
    return y_pred_proba, y_pred, roc_auc