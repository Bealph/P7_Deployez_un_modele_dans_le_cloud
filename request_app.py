import pandas as pd
import requests

# Charger les données
df = pd.read_csv("top_50_train.csv", encoding="utf-8")
df.set_index("SK_ID_CURR", inplace=True)
num_clients = df.index.unique()

# Charger le dataset vrai_val_client_data
vrai_val_client_data = pd.read_csv("top_50_vraiVal_X_train.csv", encoding='utf-8')

# Définir les headers spécifiant le type de contenu en JSON
headers = {"Content-Type": "application/json"}

def get_infos_client(selected_client: pd.DataFrame) -> tuple[float, list, list]:
    """
    Obtenez des informations sur le client sélectionné depuis l'API.

    Parameters:
    - selected_client (pandas.DataFrame): Les données du client sélectionné.

    Returns:
    - prediction_proba (float): La probabilité prédite.
    - feature_names (list): Les noms des features.
    - feature_importance (list): L'importance des features.
    """
    client_data = selected_client.to_dict(orient="records")
    try:
        response = requests.post(
            "https://alphafinance-b31189df9933.herokuapp.com/api/infos_client",
            headers=headers,
            json=client_data,
        )
        if response.status_code == 200:
            result = response.json()
            prediction_proba = result["proba"]
            feature_names = result["feature_names"]
            feature_importance = result["feature_importance"]
            return prediction_proba, feature_names, feature_importance
        else:
            print("Erreur de réponse:", response.status_code)
    except requests.exceptions.RequestException as e:
        print("Une erreur s'est produite lors de l'envoi de la requête:", e)
    return None, [], []

def get_index_and_values_from_vrai_val_client_data() -> tuple[list, list]:
    """
    Récupère les index et les valeurs du dataset vrai_val_client_data.

    Returns:
    - indexes (list): Les index des clients.
    - values (list): Les valeurs des clients.
    """
    indexes = vrai_val_client_data.index.tolist()
    values = vrai_val_client_data.to_dict(orient='records')
    return indexes, values

if __name__ == "__main__":
    # Obtenir les index et les valeurs du dataset vrai_val_client_data
    indexes, values = get_index_and_values_from_vrai_val_client_data()

    print("Index des clients:")
    print(indexes[:20])

    print("Valeurs des clients:")
    print(values[:20])

    # Sélectionner un client aléatoire
    selected_client = df.sample(1)

    # Obtenir les prédictions du client sélectionné
    prediction_proba, feature_names, feature_importance = get_infos_client(
        selected_client
    )

    print("Client sélectionné:")
    print( )

    print("Prédictions:")
    print( )

    print("Noms des features:")
    print( )

    print("Importance des features:")
 
