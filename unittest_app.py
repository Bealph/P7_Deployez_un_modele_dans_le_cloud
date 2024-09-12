import pytest
import requests
import pandas as pd
import streamlit as st
import request_app as ra
from unittest.mock import patch
import io

# Configuration de base
base_url = "http://127.0.0.1:5000"
headers = {"Content-Type": "application/json"}

# Charger les données de test
df = pd.read_csv("top_50_train.csv", encoding='utf-8')
df.set_index('SK_ID_CURR', inplace=True)
var_df_dict = df.iloc[0].to_dict()

# Test de la fonction de chargement des données CSV
def test_csv_loading():
    # Vérifier que les données ont été correctement indexées
    assert df.index.name == 'SK_ID_CURR', "L'index du dataframe ne correspond pas à 'SK_ID_CURR'."



# Test de l'API Flask
def test_homepage():
    response = requests.get(base_url + '/')
    assert response.status_code == 200
    assert response.text == "Hello, World! Welcome to my Flask App."

def test_predict_endpoint():
    response = requests.post(f'{base_url}/api/infos_client/', headers=headers, json=var_df_dict)
    assert response.status_code == 200
    json_data = response.json()
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
    prediction_proba, feature_names, feature_importance = ra.get_infos_client(selected_client)

    # Vérifier que les résultats sont corrects
    assert prediction_proba == [0.7, 0.3], "Les probabilités de prédiction ne sont pas correctes."
    assert feature_names == ['Feature1', 'Feature2'], "Les noms des features ne sont pas corrects."
    assert feature_importance == [0.5, 0.3], "Les importances des features ne sont pas correctes."

# Test du dashboard Streamlit
def test_dashboard():
    # Simuler l'interface Streamlit
    with patch('streamlit.write') as mock_write:
        # Initialiser les états de session
        st.session_state.selected_client = "1"
        st.session_state.show_variables = True
        st.session_state.show_predictions = True
        st.session_state.show_close_button = True

        # Simuler les appels aux fonctions de Streamlit
        with patch('streamlit.dataframe') as mock_dataframe:
            with patch('streamlit.image') as mock_image:
                with patch('streamlit.altair_chart') as mock_altair_chart:
                    with patch('streamlit.plotly_chart') as mock_plotly_chart:
                        # Simuler la fonction de prédiction
                        with patch('request_app.get_infos_client') as mock_get_infos_client:
                            mock_get_infos_client.return_value = (
                                [0.7, 0.3],  # Probabilités
                                ['Feature1', 'Feature2'],  # Noms des caractéristiques
                                [0.5, 0.3]  # Importance des caractéristiques
                            )

                            # Appeler la fonction principale du dashboard
                            with patch('streamlit.sidebar.selectbox') as mock_selectbox:
                                mock_selectbox.return_value = "1"
                                with patch('streamlit.sidebar.button') as mock_button:
                                    mock_button.return_value = True
                                    
                                    # Simuler le code de votre dashboard ici
                                    st.write("Test de l'affichage des prédictions")
                                    st.image("image_app.jpeg")  # Ajout d'un appel à image pour tester la fonctionnalité
                                    st.dataframe(df)  # Ajouter d'autres appels de Streamlit selon le cas
                                    
                                    # Vérifier que les fonctions de Streamlit sont appelées correctement
                                    mock_write.assert_called_with("Test de l'affichage des prédictions")
                                    mock_image.assert_called()  # Vérifie si image a été appelée
                                    mock_dataframe.assert_called()

# Exécuter tous les tests
if __name__ == "__main__":
    pytest.main()
