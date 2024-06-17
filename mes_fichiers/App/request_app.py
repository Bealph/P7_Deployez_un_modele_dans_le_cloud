import pandas as pd
import requests
#----------------------------------------------------------------

# Chargeons les données
df = pd.read_csv("top_50_train.csv", encoding='utf-8')

df.set_index('SK_ID_CURR', inplace=True)

num_client = df.index.unique()

# Définir les headers spécifiant le type de contenu en JSON
headers = {
    "Content-Type": "application/json",
}


# Envoi de la requête HTTP au serveur Flask

def get_infos_client(selected_client):
    client_data = selected_client.to_dict()
    prediction_proba = None
    feature_names = None
    feature_importance = None

    try:
        response = requests.post("http://0.0.0.0:5000/api/infos_client", headers=headers, json=client_data)

        if response.status_code == 200:
            result = response.json()
            prediction_proba = result['proba']
            feature_names = result['feature_names']
            feature_importance = result['feature_importance']

    except requests.exceptions.RequestException as e:
        print("Une erreur s'est produite lors de l'envoi de la requête:", e)

    return prediction_proba, feature_names, feature_importance