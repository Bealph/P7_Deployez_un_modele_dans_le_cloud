import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import request_app as ra
import pickle
from PIL import Image
import pandas as pd

# Import des shap_values et expected_value
with open('shap_values.pkl', 'rb') as shap_file:
    shap_values = pickle.load(shap_file)

with open('expected_value.pkl', 'rb') as expected_value_file:
    expected_value = pickle.load(expected_value_file)

# Charger l'image
image = Image.open('image_app.jpeg')
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.image(image, width=250, output_format="JPEG")


st.markdown(
    '<h1 class="header">Tableau de bord en temps réel</h1>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="centered-text">Cette application vise à fournir un système de scoring avancé permettant\
        l\'accès aux informations essentielles des clients. Grâce à des modèles prédictifs, elle offre la possibilité\
            de générer des predictions pertinentes en temps réel.<br> <br> L\'objectif principale est de faciliter \
                la prise de décision en fournissant des analyses détaillées basées sur les données des clients.</p>',
    unsafe_allow_html=True
)

# Charger les données depuis le fichier CSV
client_data = pd.read_csv("top_50_train.csv", encoding='utf-8')

vrai_val_client_data = pd.read_csv("top_50_vraiVal_X_train.csv", encoding='utf-8')

descrip_columns = pd.read_csv('HomeCredit_columns_description_translated.csv', encoding='utf-8')
lexique = descrip_columns[['Row', 'Description']]

# Initialiser les états de session si non déjà présents
if 'selected_client' not in st.session_state:
    st.session_state.selected_client = ""
if 'show_variables' not in st.session_state:
    st.session_state.show_variables = False
if 'show_predictions' not in st.session_state:
    st.session_state.show_predictions = False
if 'show_close_button' not in st.session_state:
    st.session_state.show_close_button = False

# Menu déroulant pour sélectionner un client
selected_client = st.sidebar.selectbox(
    "Sélectionnez un client",
    [""] + client_data['SK_ID_CURR'].astype(str).tolist(),
    index=0 if st.session_state.selected_client == "" else client_data['SK_ID_CURR'].astype(str).tolist().index(st.session_state.selected_client) + 1
)

# Mettre à jour l'état du client sélectionné
st.session_state.selected_client = selected_client

# Affichage des boutons
show_predictions = st.sidebar.button("Affichage des prédictions probables")
show_variables = st.sidebar.button("Afficher les 10 variables importantes")

if show_variables or show_predictions:
    st.session_state.show_close_button = True

if st.session_state.show_close_button:
    if st.sidebar.button("Fermer"):
        st.session_state.selected_client = ""
        st.session_state.show_variables = False
        st.session_state.show_predictions = False
        st.session_state.show_close_button = False

# Vérifier si un client a été sélectionné
if st.session_state.selected_client:
    if show_predictions:
        st.markdown('<style>.header, .centered-text, img {display: none;}</style>', unsafe_allow_html=True)

        try:
            if st.session_state.selected_client in client_data['SK_ID_CURR'].astype(str).values:

                # Obtenir l'indice du client sélectionné
                client_index = client_data[client_data['SK_ID_CURR'] == int(st.session_state.selected_client)].index[0]

                # Filtrer les données du client
                data_by_client = client_data.loc[client_index].drop(labels='SK_ID_CURR')
                data_by_client_df = pd.DataFrame(data_by_client).T

                prediction_proba, feature_names, feature_importance = ra.get_infos_client(data_by_client_df)
                probability = prediction_proba[0]  # La probabilité de défaut de paiement

                st.markdown('<u><h3>Évaluation du Risque de Crédit :</h3></u>', unsafe_allow_html=True)
            

                # Ajouter les phrases en fonction de la probabilité
                if probability < 0.40:
                    st.markdown('<p style="color: green; text-align: center;">Le prêt peut être accordé</p>', unsafe_allow_html=True)
                elif 0.41 <= probability <= 0.59:
                    st.markdown('<p style="color: orange; text-align: center;">Client à risque</p>', unsafe_allow_html=True)
                elif probability > 0.60:
                    st.markdown('<p style="color: red; text-align: center;">Le prêt ne peut pas être accordé</p>', unsafe_allow_html=True)


                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    title={'text': "Probabilité de défaut de paiement"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 40], 'color': "lightgreen"},
                            {'range': [40, 60], 'color': "orange"},
                            {'range': [60, 100], 'color': "red"}
                        ],
                    }
                ))

                st.plotly_chart(fig_gauge, use_container_width=True)

                st.write('------------------------------')

                ######################## ChartPlot ##############################

                st.markdown('<u><h3>Analyse des Probabilités par Décision :</h3></u>', unsafe_allow_html=True)
                classes = ['Risque', 'Favorable']
                values = [prediction_proba[0], prediction_proba[1]]

                # Création du graphique avec couleurs personnalisées
                fig_chartplot = px.bar(
                    x=classes,
                    y=values,
                    labels={'x': 'Décision', 'y': 'Probabilité'},
                    color=classes,  # Utilisation des classes pour colorier les barres
                    color_discrete_map={'Risque': 'red', 'Favorable': 'green'}  # Spécification des couleurs
                )

                # Affichage du graphique
                st.plotly_chart(fig_chartplot, use_container_width=True)

                st.write('------------------------------')

                st.markdown('<u><h3>Impact des Facteurs Clés sur le Score de Crédit du Client :</h3></u>', unsafe_allow_html=True)

                prediction_proba, feature_names, feature_importance = ra.get_infos_client(pd.DataFrame(data_by_client).T)

                feature_names_upper = [name.upper() for name in feature_names]

                top_10_indices = sorted(range(len(feature_importance)), key=lambda i: feature_importance[i], reverse=True)[:10]
                top_10_features = [(feature_names_upper[i], feature_importance[i]) for i in top_10_indices]

                top_10_df = pd.DataFrame(top_10_features, columns=['Variables', 'Importance'])
                top_10_df['Variables'] = top_10_df['Variables'].str.lower()
                var_list = top_10_df['Variables'].tolist()

                # Extraire les shap_values correspondant au client sélectionné
                client_index = client_data[client_data['SK_ID_CURR'] == int(st.session_state.selected_client)].index[0]
                shap_values_client = shap_values[client_index]
                
                # Assurez-vous que les noms de variables dans `filtered_feature` correspondent à ceux dans `shap_values_client`
                shap_values_client_subset = {var: shap_values_client[feature_names.index(var)] for var in var_list}
                shap_df = pd.DataFrame(shap_values_client_subset, index=[0])

                st.write("Les valeurs SHAP positives ou négatives indiquent l'ampleur de l'impact, tandis que les caractéristiques les plus à droite sont les plus influentes.")

                c = alt.Chart(shap_df.melt()).mark_bar().encode(
                    x=alt.X('variable:N', title='Feature'),
                    y=alt.Y('value:Q', title='SHAP Value'),
                    color=alt.Color('value:Q', scale=alt.Scale(scheme='viridis'), title='SHAP Value'),
                    tooltip=['variable:N', 'value:Q']
                ).properties(
                    title=f'Pour le Client {st.session_state.selected_client}',
                    width=600,
                    height=400
                )

                st.altair_chart(c)

                st.write('------------------------------')

                ######################## Decision Plot ##############################REVOIR CETTE PARTIE : verifier l'expected_value !!!

                st.markdown('<u><h3>Comparaison des Caractéristiques de Décision avec la Valeur Attendue :</h3></u>', unsafe_allow_html=True)
                val_expected_value = expected_value
                expected_values_list = [val_expected_value]
                expected_value_avg = round(sum(expected_values_list) / len(expected_values_list), 3)

                shap_df_selected_melted = shap_df.melt()
                shap_df_selected_melted['is_expected_value'] = 'No'
                shap_df_selected_melted.loc[shap_df_selected_melted['variable'] == 'expected_value', 'is_expected_value'] = 'Yes'

                decision_chart = alt.Chart(shap_df_selected_melted, title="Decision Plot Features").mark_line(opacity=0.3).encode(
                    y='variable:N',
                    x='value:Q',
                    detail='index:N',
                    color=alt.Color('is_expected_value:N', title='Expected Value', scale=alt.Scale(domain=['Yes', 'No'], range=['green', 'red']))
                ).properties(
                    title=f'Pour le Client {st.session_state.selected_client}',
                    width=600,
                    height=400
                )

                expected_value_rule = alt.Chart(pd.DataFrame({'expected_value': [expected_value_avg]})).mark_rule(strokeDash=[2, 2]).encode(
                    y='expected_value:Q',
                    color=alt.value('green')
                )

                st.altair_chart(decision_chart + expected_value_rule)

                st.write("Ce qu'il faut observer :")
                st.write(" " + "Les zones où la ligne de la caractéristique traverse la ligne de règle pointillée indiquent des changements significatifs dans la décision.")
                st.write(" " + "En analysant les intersections et les pentes de ces lignes, on peut identifier les caractéristiques clés ayant le plus d'impact sur la décision de crédit.")

            else:
                st.error("Le client sélectionné est introuvable dans les données.")

        except KeyError as e:
            st.error(f"Erreur lors de l'accès aux données : {e}")

    if show_variables:
        st.markdown('<style>.header, .centered-text, img {display: none;}</style>', unsafe_allow_html=True)

        try:
            if st.session_state.selected_client in client_data['SK_ID_CURR'].astype(str).values:

                # Obtenir l'indice du client sélectionné
                client_index = client_data[client_data['SK_ID_CURR'] == int(st.session_state.selected_client)].index[0]

                # Filtrer les données du client
                data_by_client = client_data.loc[client_index].drop(labels='SK_ID_CURR')
                data_by_client_df = pd.DataFrame(data_by_client).T

                prediction_proba, feature_names, feature_importance = ra.get_infos_client(data_by_client_df)

                if feature_names is None or feature_importance is None:
                    st.error("Erreur : Les données de l'API ne sont pas disponibles.")
                else:
                    feature_names_upper = [name.upper() for name in feature_names]

                    top_10_indices = sorted(range(len(feature_importance)), key=lambda i: feature_importance[i], reverse=True)[:10]
                    top_10_features = [(feature_names_upper[i], feature_importance[i]) for i in top_10_indices]

                    top_10_df = pd.DataFrame(top_10_features, columns=['Row', 'Importance'])
                    top_10_lexique = pd.merge(top_10_df, lexique, on='Row')

                    fig, ax = plt.subplots()


                    st.markdown("<u><h3>Top 10 des Variables Clés pour l'Octroi de Crédit et Leur Description:</h3></u>", unsafe_allow_html=True)

                    st.table(top_10_lexique[['Row', 'Description']].rename(columns={'Row': 'Variables', 'Description': 'Lexique'}))

                    ####################### Histogramme ###########################

                    st.markdown("<u><h3>Classement des 10 Variables Impactant les Décisions de Crédit :</h3></u>", unsafe_allow_html=True)
                    
                    st.write(" ")

                    bar_plot = alt.Chart(top_10_df).mark_bar().encode(
                        x=alt.X('Row', title='Variables', sort='-y'),
                        y=alt.Y('Importance', title='Importance')
                    ).properties(width=500, height=400)

                    st.altair_chart(bar_plot)

                    st.write('------------------------------')

                    ####################### Boxplot ###########################     

                    st.markdown("<u><h3>Analyse des Principales Caractéristiques de Crédit pour le Client Sélectionné :</h3></u>", unsafe_allow_html=True)
                    
                    st.write(" ")

                    feature_info = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': feature_importance
                    }).sort_values(by='Importance', ascending=False)

                    top_10_features = feature_info.head(10)
                    top_10_features_names = top_10_features['Feature'].tolist()
                    top_10_client_data = client_data[top_10_features_names]
                    filtered_client_data = data_by_client[data_by_client.index.isin(top_10_features_names)]

                    client_points = pd.DataFrame({
                        'Feature': filtered_client_data.index,
                        'Value': filtered_client_data.values,
                        'Type': [f"Client : {st.session_state.selected_client}"] * len(filtered_client_data)
                    })

                    boxplot = alt.Chart(top_10_client_data.melt()).mark_boxplot().encode(
                        x='variable:O',
                        y='value:Q'
                    ).properties(width=600, height=400)

                    boxplot_by_client = alt.Chart(pd.concat([top_10_client_data, client_points])).mark_point(
                        color='red',
                        size=100,
                        filled=True
                    ).encode(
                        x='Feature:N',
                        y='Value:Q',
                        shape='Type:N'
                    ).properties(width=600, height=400)

                    combined_chart = boxplot + boxplot_by_client
                    st.altair_chart(combined_chart, use_container_width=True)

                    # Obtenir les index et les valeurs du dataset vrai_val_client_data
                    indexes, values = ra.get_index_and_values_from_vrai_val_client_data()

                    # Trouver l'index du client sélectionné
                    client_index = int(st.session_state.selected_client)

                    # Trouver les données du client
                    client_data_by_id = vrai_val_client_data[vrai_val_client_data['SK_ID_CURR'] == client_index]

                    if not client_data_by_id.empty:
                        # Réinitialiser l'index pour qu'il devienne une colonne normale
                        client_data_by_id = client_data_by_id.reset_index(drop=True)

                        # Appliquer le style pour formater les valeurs et centrer le texte
                        styled_df = client_data_by_id.style.format("{:.1f}")

                        # Afficher le DataFrame formaté sans la colonne d'index
                        st.write("Données du client sélectionné:")
                        st.dataframe(styled_df)

                       
                    else:
                        st.error("Les données du client sélectionné ne sont pas disponibles.")

                    st.write('------------------------------')

                    ###################### chartplot ####################                    

                    st.markdown("<u><h3>Importance des caractéristiques pour le Modèle de Crédit :</h3></u>", unsafe_allow_html=True)
                    st.write("La position et la longueur de chaque barre reflètent respectivement l'importance de chaque caractéristique et son impact sur la décision d'accorder un crédit.")

                    fig_chartplot = go.Figure(go.Bar(y=feature_names, x=feature_importance, orientation='h', marker_color='skyblue'))
                    fig_chartplot.update_layout(xaxis_title='Importance', yaxis_title='Caractéristiques')

                    # Rafraîchissement du graphique
                    st.plotly_chart(fig_chartplot, use_container_width=True)

                    st.write('------------------------------')

                    #####################  Decision Plot  #####################

                    st.markdown("<u><h3>Poids des Critères dans l'Évaluation de Crédit :</h3></u>", unsafe_allow_html=True)
                    
                    st.write(" Les caractéristiques qui contribuent le plus à l'accord de crédit sont visualisées par des barres plus hautes, tandis que celles ayant une influence moindre sont représentées par des barres plus courtes.")

                    fig, ax = plt.subplots()

                    fig_decision_plot = go.Figure(go.Bar(x=feature_names, y=feature_importance, 
                                            hoverinfo='x+y', marker=dict(color='skyblue')))
                    fig_decision_plot.update_layout(xaxis_title='Caractéristiques', yaxis_title='Importance')

                    st.plotly_chart(fig_decision_plot)
            else:
                st.error("Le client sélectionné est introuvable dans les données.")

        except KeyError as e:
            st.error(f"Erreur lors de l'accès aux données : {e}")
