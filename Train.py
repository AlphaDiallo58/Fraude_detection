# Train.py

import lightgbm as lgb
import xgboost as xgb
from sklearn.utils import class_weight

def train_model(model_name, param, X_train, y_train, n_estimators=1000, apply_class_weight=True):
    """
    Entraînement (LightGBM ou XGBoost) avec les hyperparamètres donnés.

    """
    if apply_class_weight:
        class_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)
        scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    else:
        class_weights = None
        scale_pos_weight = 1

    param['scale_pos_weight'] = scale_pos_weight  # Appliquer le poids des classes dans les paramètres
    
    if model_name == 'LGBM':
        # LightGBM
        train_data = lgb.Dataset(X_train, label=y_train, weight=class_weights) if class_weights is not None else lgb.Dataset(X_train, label=y_train)
        model = lgb.train(param, train_data, num_boost_round=n_estimators)

    elif model_name == 'XGBoost':
        # XGBoost
        model = xgb.XGBClassifier(**param, n_estimators=n_estimators)
        model.fit(X_train, y_train, sample_weight=class_weights)

    else:
        raise ValueError(f"Modèle inconnu : {model_name}. Choisissez 'LGBM' ou 'XGBoost'.")
    
    return model
