import pytest
import requests
import pandas as pd

base_url = "http://127.0.0.1:5000"

# Charger les données de test
df = pd.read_csv("top_50_train.csv", encoding='utf-8')
df.set_index('SK_ID_CURR', inplace=True)

# Sélectionner un client de test
var_df_dict = df.iloc[0].to_dict()

# Headers pour les requêtes
headers = {
    "Content-Type": "application/json",
}

# Test pour la route de base
def test_homepage():
    response = requests.get(base_url + '/')
    assert response.status_code == 200
    assert response.text == "Hello, World! Welcome to my Flask App."

# Test du endpoint de prédiction
def test_predict_endpoint():
    response = requests.post(f'{base_url}/api/infos_client/', headers=headers, json=var_df_dict)
    assert response.status_code == 200
    json_data = response.json()
    assert 'proba' in json_data
    assert 'feature_names' in json_data
    assert 'feature_importance' in json_data

# Test avec des données manquantes
def test_predict_with_missing_data():
    # Enlever une colonne importante pour simuler des données incomplètes
    incomplete_data = var_df_dict.copy()
    incomplete_data.pop('AMT_INCOME_TOTAL', None)
    
    response = requests.post(f'{base_url}/api/infos_client/', headers=headers, json=incomplete_data)
    assert response.status_code != 200  # Ici on s'attend à une erreur

# Test avec des données incorrectes
def test_predict_with_invalid_data():
    # Simuler des données incorrectes (par exemple, en mettant des strings à la place de valeurs numériques)
    invalid_data = var_df_dict.copy()
    invalid_data['AMT_INCOME_TOTAL'] = "invalid_value"
    
    response = requests.post(f'{base_url}/api/infos_client/', headers=headers, json=invalid_data)
    assert response.status_code != 200  # Ici on s'attend à une erreur aussi
