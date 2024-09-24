import pytest
import pandas as pd
from unittest.mock import patch
from app import app
from request_app import get_infos_client

# Charger les données de test
df = pd.read_csv("top_50_train.csv", encoding='utf-8')
df.set_index('SK_ID_CURR', inplace=True)
var_df_dict = df.iloc[0].to_dict()

@pytest.fixture()
def client():
    with app.test_client() as client:
        yield client

# Test de la fonction de chargement des données CSV
def test_csv_loading():
    # Vérifier que les données ont été correctement indexées
    assert df.index.name == 'SK_ID_CURR', "L'index du dataframe ne correspond pas à 'SK_ID_CURR'."

# Test de la route principale
def test_homepage(client):
    response = client.get('/')
    assert response.status_code == 200
    assert response.data.decode() == "Hello, World! Welcome to my Flask App."

# Test de l'API de prédiction
def test_predict_endpoint(client):
    response = client.post('/api/infos_client/', json=var_df_dict)
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'proba' in json_data
    assert 'feature_names' in json_data
    assert 'feature_importance' in json_data

# Test de la fonction de prédiction de l'API
@patch('requests.post')
def test_prediction_function(mock_post):
    # Configurer le comportement simulé de la requête
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        'proba': [0.7, 0.3],
        'feature_names': ['Feature1', 'Feature2'],
        'feature_importance': [0.5, 0.3]
    }

    # Appeler la fonction de prédiction
    selected_client = df.sample(1)
    prediction_proba, feature_names, feature_importance = get_infos_client(selected_client)

    # Vérifier que les résultats sont corrects
    assert prediction_proba == [0.7, 0.3], "Les probabilités de prédiction ne sont pas correctes."
    assert feature_names == ['Feature1', 'Feature2'], "Les noms des features ne sont pas corrects."
    assert feature_importance == [0.5, 0.3], "Les importances des features ne sont pas correctes."


# Exécuter tous les tests
if __name__ == "__main__":
    pytest.main()
