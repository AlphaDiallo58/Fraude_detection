# data_processing_n_analysis.py

import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Fonction pour visualiser la distribution de 'Time'
def plot_time_distribution(df):
    sns.histplot(df['Time'], kde=True, bins=50)
    plt.title("Distribution de la colonne Time")
    plt.show()

# Fonction pour visualiser la relation entre 'Time' et 'Class'
def plot_time_class_relation(df):
    sns.boxplot(x=df['Class'], y=df['Time'])
    plt.title("Relation entre Time et les classes (fraude vs non-fraude)")
    plt.show()

# Fonction pour ajouter la colonne 'Hour'
def add_hour_column(df):
    df['Hour'] = (df['Time'] % 86400) // 3600

# Fonction pour visualiser les transactions par heure
def plot_hourly_transactions(df):
    fraud_hourly = df[df['Class'] == 1]['Hour']
    non_fraud_hourly = df[df['Class'] == 0]['Hour']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Histogramme des transactions non-frauduleuses
    sns.histplot(non_fraud_hourly, bins=24, kde=True, ax=axes[0], color='blue')
    axes[0].set_title("Transactions non-frauduleuses par heure")
    axes[0].set_xlabel("Heure")
    axes[0].set_ylabel("Nombre de transactions")

    # Histogramme des transactions frauduleuses
    sns.histplot(fraud_hourly, bins=24, kde=True, ax=axes[1], color='red')
    axes[1].set_title("Transactions frauduleuses par heure")
    axes[1].set_xlabel("Heure")

    plt.tight_layout()
    plt.show()

# Fonction pour créer la heatmap de corrélation
def plot_correlation_heatmap(df):
    columns_v = [f'V{i}' for i in range(1, 29)]
    data_v = df[columns_v]
    correlation_matrix = data_v.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", cbar=True, square=True)
    plt.title('Heatmap des Corrélations (V1 à V28)', fontsize=16)
    plt.tight_layout()
    plt.show()

# Fonction pour prétraiter les données et appliquer le rééchantillonnage
def data_modeling(df):

    scaler = MinMaxScaler()
    X = df.drop(columns=['Class', 'Hour', 'Time'])
    Amount_normalized = scaler.fit_transform(X[['Amount']])
    X['Amount'] = Amount_normalized

    y = df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    
    return X_train, X_test, y_train, y_test, X_train_resampled, y_train_resampled
