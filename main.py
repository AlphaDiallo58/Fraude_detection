import pandas as pd
from Train import train_model
from Test import test_model
from data_processing_n_analysis import data_modeling,plot_time_distribution, plot_time_class_relation, add_hour_column, plot_hourly_transactions, plot_correlation_heatmap


def main():
    df = pd.read_csv('creditcard.csv')

    # Visualisations
    plot_time_distribution(df)
    plot_time_class_relation(df)
    add_hour_column(df)
    plot_hourly_transactions(df)
    plot_correlation_heatmap(df)

    # Paramètres pour LGBM
    params_lgb = {
    'objective': 'binary',
    'metric': 'binary_error',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'verbose': -1                 
    }

    # Paramètres pour XGBoost
    params_xgb = {
    'objective': 'binary:logistic',  
    'eval_metric': 'logloss',       
    'learning_rate': 0.05,           
    'max_depth': 6,                  
    'subsample': 0.8,               
    'colsample_bytree': 0.8,        
    'verbosity': 1  
    }   


    X_train, X_test, y_train, y_test, X_train_resampled, y_train_resampled = data_modeling(df)

    # Entraîner les modèles avec poids
    lgb_model_weighted = train_model('LGBM', params_lgb, X_train, y_train, apply_class_weight=True)
    xgb_model_weighted = train_model('XGBoost', params_xgb, X_train, y_train, apply_class_weight=True)

    # Entraîner les modèles sans poids (sur les données rééchantillonnées)
    lgb_model_not_weighted = train_model('LGBM', params_lgb, X_train_resampled, y_train_resampled, n_estimators=1000, apply_class_weight=False)

    # Tester les modèles
    test_model(lgb_model_weighted, X_test, y_test)
    test_model(xgb_model_weighted, X_test, y_test)
    test_model(lgb_model_not_weighted, X_test, y_test)

if __name__ == "__main__":
    main()