from flask import Flask, jsonify, request
import pandas as pd
import pickle

app = Flask(__name__)
app.config['TESTING'] = True

# Chemin vers le fichier pickle contenant le modèle
chemin_modele = "mon_best_modele_entraine_LightGBM.pkl"

with open(chemin_modele, 'rb') as fichier:
    print("utilisation du meilleur modele : lightGBM")
    modele = pickle.load(fichier)


@app.route('/')
def hello_world():
    return 'Hello, World! Welcome to my Flask App.'

# Routes Flask
@app.route('/api/infos_client/', methods=['POST'])
def data_client():
    # Obtenir les données transmises par la requête
    data = request.get_json()

    df_data = pd.DataFrame([data])

    # Faire la prédiction probable à partir du modele entrainé
    prediction_proba = modele.predict_proba(df_data)

    # Obtenir les noms des features à partir des colonnes du DataFrame
    feature_names = df_data.columns.tolist()
    
    # Obtenir les feature importances à partir du modele entrainé
    feature_importance = modele.feature_importances_.tolist()

    # Préparer la réponse de la requête en incluant les nom des feature et leurs importances
    response = {
        'proba': prediction_proba[0].tolist(),
        'feature_names': feature_names,
        'feature_importance': feature_importance
    }

    return jsonify(response)

if __name__ == '__main__':
    # Lancement de l'application Flask
    app.run(debug=True)