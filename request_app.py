import pandas as pd
import requests

# Chargeons les données
df = pd.read_csv("top_50_train.csv", encoding="utf-8")
df.set_index("SK_ID_CURR", inplace=True)
num_clients = df.index.unique()

# Définir les headers spécifiant le type de contenu en JSON
headers = {"Content-Type": "application/json"}


def get_infos_client(selected_client: pd.DataFrame) -> tuple[float, list, list]:
    """
    Get information about the selected client from the API.

    Parameters:
    - selected_client (pandas.DataFrame): The selected client data.

    Returns:
    - prediction_proba (float): The predicted probability.
    - feature_names (list): The names of the features.
    - feature_importance (list): The importance of the features.
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
    except requests.exceptions.RequestException as e:
        print("Une erreur s'est produite lors de l'envoi de la requête:", e)


if __name__ == "__main__":
    # Sélectionner un client aléatoire
    selected_client = df.sample(1)

    # Obtenir les prédictions du client sélectionné
    prediction_proba, feature_names, feature_importance = get_infos_client(
        selected_client
    )

    print("Client sélectionné:")
    print(selected_client)

    print("Prédictions:")
    print(prediction_proba)

    print("Noms des features:")
    print(feature_names)

    print("Importance des features:")
    print(feature_importance)
