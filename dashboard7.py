# Importation des bibliothèques Dash et Plotly

# -------------------------------------------------------------------------
# pip install dash
# python.exe -m pip install --upgrade pip
# pip install  pandas matplotlib pillow requests scikit-learn  plotly dash
# -------------------------------------------------------------------------

from dash import Dash, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import shap
# import matplotlib.pyplot as plt

# from PIL import Image
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import requests
import plotly.graph_objects as go
import plotly.express as px
import pickle


# Initialisation de l'application Dash
app = Dash(__name__, suppress_callback_exceptions=True)
# app = dash.Dash(__name__)

# URL de l'API
API_URL = "http://3.84.177.36:8000/" #  "http://127.0.0.1:8000/"  "https://votre-url-d-api.herokuapp.com/"

# Chargement des données
import os

os.chdir('C:/Users/Fane0763/OpenClassroom/OC Projet 7')
data_train = pd.read_csv('./out_put/train_df.csv').set_index('SK_ID_CURR')
data_test  = pd.read_csv('./out_put/test_df.csv').set_index('SK_ID_CURR')
X_train    = pd.read_csv('./out_put/X_train.csv').set_index('SK_ID_CURR')

X_train = pd.read_csv('./out_put/X_train.csv').set_index('SK_ID_CURR')
cols = X_train.select_dtypes(['float64']).columns
scaler = StandardScaler()
scaler.fit(X_train[cols])

# data_train_scaled = pd.read_csv('./out_put/X_train_std.csv').set_index('SK_ID_CURR')
# data_test_scaled = pd.read_csv('./out_put/X_test_std.csv').set_index('SK_ID_CURR')

listvar = X_train.columns.tolist()

# Sélection des colonnes numériques pour la mise à l'échelle
data_test_scaled = data_test[listvar].copy()
data_test_scaled[cols] = scaler.transform(data_test_scaled[cols])

data_train_scaled = data_train[listvar].copy()
data_train_scaled[cols] = scaler.transform(data_train[cols])


# Chargement du modèle et des données
model = pickle.load(open('C:/Users/Fane0763/OpenClassroom/OC Projet 7/Models/best_LGBMClassifier.pkl', 'rb'))

# Initialisation de l'explainer Shapley pour les valeurs locales
explainer = shap.TreeExplainer(model)

# Fonction de prétraitement des données
def preprocessing(df, scaler_type):
    cols = df.select_dtypes(['float64']).columns
    df_scaled = df.copy()
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    df_scaled[cols] = scaler.fit_transform(df[cols])
    return df_scaled


# Fonction pour obtenir la prédiction
def get_prediction(client_id):
    url_get_pred = API_URL + "prediction/" + str(client_id)
    response = requests.get(url_get_pred)
    proba_default = round(float(response.content), 3)
    best_threshold = 0.5
    decision = "Refusé" if proba_default >= best_threshold else "Accordé"
    return proba_default, decision

# def get_prediction(client_id):
#     try:
#         url_get_pred = API_URL + "prediction/" + str(client_id)
#         response = requests.get(url_get_pred)
#         response.raise_for_status()  # Lève une exception si la requête échoue (status code >= 400)
#         proba_default = round(float(response.content), 3)
#         best_threshold = 0.5
#         decision = "Refusé" if proba_default >= best_threshold else "Accordé"
#         return proba_default, decision
#     except Exception as e:
#         print(f"Erreur lors de la récupération des données de l'API : {e}")
#         return None, "Erreur lors de la récupération des données"


# Fonction pour afficher la jauge de score
def plot_score_gauge(proba):
    fig = go.Figure(go.Indicator(
        domain={'x': [0, 1], 'y': [0, 1]},
        value=proba * 100,
        mode="gauge+number+delta",
        title={'text': "Jauge de score"},
        delta={'reference': 54},
        gauge={'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "black"},
               'bar': {'color': "MidnightBlue"},
               'steps': [
                   {'range': [0, 20], 'color': "Green"},
                   {'range': [20, 45], 'color': "LimeGreen"},
                   {'range': [45, 54], 'color': "Orange"},
                   {'range': [54, 100], 'color': "Red"}],
               'threshold': {'line': {'color': "brown", 'width': 4}, 'thickness': 1,
                             'value': 54}}))
    return fig


def plot_score_gauge2(proba):
    # Créez les seuils pour les couleurs
    thresholds = [0, 20, 45, 54, 100]
    colors = ["Green", "LimeGreen", "Orange", "Red"]

    # Créez un DataFrame contenant les données pour la jauge
    df = pd.DataFrame({
        'proba': [proba * 100],
        'threshold': [50]  # Référence pour la jauge
    })

    # Créez le graphique de jauge avec Plotly Express
    fig = px.indicator(
        df,
        title="Jauge de score",
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "MidnightBlue"},
            'steps': [
                {'range': [0, thresholds[0]], 'color': colors[0]},
                {'range': [thresholds[0], thresholds[1]], 'color': colors[1]},
                {'range': [thresholds[1], thresholds[2]], 'color': colors[2]},
                {'range': [thresholds[2], thresholds[3]], 'color': colors[3]}
            ],
            'threshold': {
                'line': {'color': "brown", 'width': 4},
                'thickness': 1,
                'value': df['threshold'].iloc[0]
            }
        },
        template='plotly_dark'
    )

    # Masquez les légendes et l'échelle de couleur
    fig.update_traces(showlegend=False)
    fig.update_coloraxes(colorbar=dict(visible=False))

    return fig

# Graphique Shapley
import plotly.express as px

def generate_local_interpretation_graph(client_id):
    if client_id is None:
        raise PreventUpdate  # Si aucun client n'est sélectionné, empêchez la mise à jour du graphique

    # Obtenez l'indice correspondant au client sélectionné (assurez-vous que data_test.index est une liste d'ID clients)
    client_index = data_test.index.get_loc(client_id)

    # Sélectionnez les valeurs SHAP pour le client spécifique
    client_data = data_test_scaled[data_test_scaled.index == client_id]
    shap_values_client = explainer.shap_values(client_data)[0]

    # Créez un DataFrame contenant les caractéristiques et les valeurs SHAP pour le client
    shap_df = pd.DataFrame(data={
        'Feature': client_data.columns,  # Noms des caractéristiques
        'SHAP Value': shap_values_client[0]  # Valeurs SHAP pour le client
    })

    # Triez le DataFrame par valeurs SHAP (optionnel)
    shap_df = shap_df.sort_values(by='SHAP Value', ascending=False)

    # Créez un graphique à barres horizontales avec Plotly Express
    shap_graph = px.bar(
        shap_df,
        x='SHAP Value',
        y='Feature',
        orientation='h',
        labels={'SHAP Value': 'Valeur SHAP', 'Feature': 'Caractéristique'},
        title='Interprétation locale - Valeurs SHAP'
    )

    # Retournez le graphique SHAP pour l'onglet 'local_interpretation'
    return shap_graph




# Mise en page de l'application Dash
app.layout = html.Div([
    # Titre de la page
    html.H1("Dashboard Prêt à dépenser"),

    # Menu déroulant pour sélectionner l'ID du client
    dcc.Dropdown(
        id='client-dropdown',
        options=[{'label': str(id_client_dash), 'value': id_client_dash} for id_client_dash in data_test.index],
        value=None,  # Valeur par défaut, vous pouvez la changer si nécessaire
        placeholder='Sélectionnez un client'
    ),

    # Contenu dynamique basé sur les onglets
    dcc.Tabs(id='tabs', value='home', children=[
        dcc.Tab(label='Home', value='home'),
        dcc.Tab(label='Information du client', value='client_info'),
        dcc.Tab(label='Interprétation locale', value='local_interpretation'),
        dcc.Tab(label='Interprétation globale', value='global_interpretation')
    ]),

    # Contenu spécifique aux onglets
    html.Div(id='tab-content')
])


# Fonction de mise à jour du contenu des onglets
@app.callback(Output('tab-content', 'children'),
              [Input('tabs', 'value')])

def update_tab(tab_name):
    if tab_name == 'home':

        return html.Div([
            html.H2("Bienvenue sur le tableau de bord Prêt à dépenser"),
            html.Div([
                dcc.Markdown("Ce site contient un dashboard interactif permettant d'expliquer aux clients les "
                             "d'approbation ou refus de leur demande de crédit.\n"
                             "\nLes prédictions sont calculées à partir d'un algorithme d'apprentissage automatique "
                             "préalablement entraîné. Il s'agit d'un modèle *Light GBM* (Light Gradient Boosti"
                             "Les données utilisées sont disponibles [ici](https://www.kaggle.com/c/home-credi"
                             "Lors du déploiement, un échantillon de ces données a été utilisé.\n"
                             "\nLe dashboard est composé de plusieurs pages :\n"
                             "- **Information du client**: Vous pouvez y retrouver toutes les informations rel"
                             "selectionné dans la colonne de gauche, ainsi que le résultat de sa demande de cr"
                             "Je vous invite à accéder à cette page afin de commencer.\n"
                             "- **Interprétation locale**: Vous pouvez y retrouver quelles caractéristiques du"
                             "influencé le choix d'approbation ou refus de la demande de crédit.\n"
                             "- **Intérprétation globale**: Vous pouvez y retrouver notamment des comparaisons"
                             "les autres clients de la base de données ainsi qu'avec des clients similaires.")
            ])
        ])
    elif tab_name == 'client_info':
        # ... (contenu de la page Information du client)
        return [
            html.H2("Information du client"),
            html.Div([
                html.Button("Statut de la demande", id='start-button'),
                html.Div(id='prediction-output')
            ]),
            html.Div(id='client-info-expander-output')
        ]        

    elif tab_name == 'local_interpretation':
        # ... (contenu de la page Interprétation locale)
        # interpretation_graph = generate_local_interpretation_graph(client_id) 
        return html.Div([
            html.H2("Interprétation locale"),
            # html.Div(id='shap_graph'),
            html.Div(id='shap-graph') 
        ])
    
    elif tab_name == 'global_interpretation':
        # ... (contenu de la page Interprétation globale)
        return html.Div([
            html.H2("Interprétation globale"),
            # ... (contenu de la page Interprétation globale)
        ])
# Callback pour mettre à jour les informations du client
@app.callback(
    Output('client-info-expander-output', 'children'),
    [Input('start-button', 'n_clicks')],
    [State('client-dropdown', 'value')]
)
def update_client_info_expander(n_clicks, client_id):
    if n_clicks is None or not client_id:
        return ''
    else:
        client_info = pd.DataFrame(data_test.loc[data_test.index == client_id])
        return html.Div([
            html.Div("Voici les informations du client:"),
            dcc.Markdown(client_info.to_markdown())
        ])
            
# Fonction de mise à jour de la sortie de la prédiction
@app.callback(Output('prediction-output', 'children'),
              [Input('start-button', 'n_clicks')],
              [State('client-dropdown', 'value')])

def update_prediction_output(n_clicks, client_id):
    if n_clicks is None:
        return ''
    else:
        if client_id and client_id != '<Select>':
            probability, decision = get_prediction(client_id)
            if decision == 'Accordé':
                return html.Div([
                    html.H2("RÉSULTAT DE LA DEMANDE"),
                    html.Div("Crédit accordé", style={'color': 'green'}),
                    dcc.Graph(figure=plot_score_gauge(probability))
                ])
            else:
                return html.Div([
                    html.H2("RÉSULTAT DE LA DEMANDE"),
                    html.Div("Crédit refusé", style={'color': 'red'}),
                    dcc.Graph(figure=plot_score_gauge(probability))
                ])
        else:
            return ''

# Callback pour mettre à jour interpretabilité
# @app.callback(
#     Output('shap_graph', 'figure'),
#     [Input('client-dropdown', 'value')]
# )

@app.callback(
    Output('shap-graph', 'children'),
    [Input('client-dropdown', 'value')]
)
def update_shap_graph(client_id):
    if client_id is None:
        raise PreventUpdate  # Si aucun client n'est sélectionné, empêchez la mise à jour du graphique

    # Obtenez l'indice correspondant au client sélectionné (assurez-vous que data_test.index est une liste d'ID clients)
    # client_index = data_test.index.get_loc(client_id)
    # Sélectionnez les valeurs SHAP pour le client spécifique
    data = data_test_scaled[data_test_scaled.index == client_id]

    # Créez un graphique à barres horizontal avec Plotly Express (à titre d'exemple)
    shap_graph = px.bar(
        data,  # Remplacez 'data' par les données que vous souhaitez afficher dans le graphique
        x='Feature',  # Remplacez 'Feature' par le nom de votre caractéristique
        y='SHAP Value',  # Remplacez 'SHAP Value' par le nom de votre valeur SHAP
        orientation='h',
        labels={'SHAP Value': 'Valeur SHAP', 'Feature': 'Caractéristique'},
        title='Interprétation locale - Valeurs SHAP'
    )

    # Retournez le graphique SHAP pour l'onglet 'local_interpretation'
    return dcc.Graph(figure=shap_graph)


# @app.callback(
#     Output('shap_graph', 'children'),
#     [Input('start-button', 'n_clicks')],
#     [State('client-dropdown', 'value')]
# )
# 
# def update_shap_graph(client_id):
#     if client_id is None:
#         raise PreventUpdate  # Si aucun client n'est sélectionné, empêchez la mise à jour du graphique
# 
#     # Obtenez l'indice correspondant au client sélectionné (assurez-vous que data_test.index est une liste d'ID clients)
#     client_index = data_test.index.get_loc(client_id)
#     print(client_index)
#     # Sélectionnez les valeurs SHAP pour le client spécifique
#     client_data = data_test_scaled[data_test_scaled.index == client_id]
#     print(client_data)
#     shap_values_client = explainer.shap_values(client_data)[0][:, 1]
# 
#     # Créez un graphique SHAP interactif avec Plotly
#     shap.initjs()  # Initialisation du JavaScript pour SHAP (si ce n'est pas déjà fait)
#     shap_graph = shap.force_plot(explainer.expected_value, shap_values_client, data_test.iloc[client_index, :])
#     # shap_graph = shap.plots.force(explainer.expected_value, shap_values_client)
#     # Retournez le graphique SHAP pour l'onglet 'local_interpretation'
#     return shap_graph

# Point d'entrée de l'application Dash
if __name__ == '__main__':
    app.run_server(debug=True, # port=9000
                   )
