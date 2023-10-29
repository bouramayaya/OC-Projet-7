# Importation des bibliothèques Dash et Plotly

# -------------------------------------------------------------------------
# pip install dash
# python.exe -m pip install --upgrade pip
# pip install  pandas matplotlib pillow requests scikit-learn  plotly dash
# -------------------------------------------------------------------------

from dash import Dash, dcc, html, Input, Output, State
# import shap
# import matplotlib.pyplot as plt

# from PIL import Image
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import requests
import plotly.graph_objects as go
import plotly.express as px

# Initialisation de l'application Dash
app = Dash(__name__, suppress_callback_exceptions=True)
# app = dash.Dash(__name__)

# URL de l'API
API_URL = "http://127.0.0.1:9000/"  # "https://votre-url-d-api.herokuapp.com/"

# Chargement des données
import os

os.chdir('C:/Users/Fane0763/OpenClassroom/OC Projet 7')
data_train = pd.read_csv('./out_put/train_df.csv').set_index('SK_ID_CURR')
data_test = pd.read_csv('./out_put/test_df.csv').set_index('SK_ID_CURR')


# ... (autres fonctions et prétraitements)
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

                # Callback pour mettre à jour le résultat de la demande
        @app.callback(
            Output('prediction-output', 'children'),
            [Input('start-button', 'n_clicks')],
            [State('client-dropdown', 'value')]
        )
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
                            html.Div("Crédit refusé", style={'color': 'red'})
                        ])
                else:
                    return ''

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

    elif tab_name == 'local_interpretation':
        # ... (contenu de la page Interprétation locale)
        return html.Div([
            html.H2("Interprétation locale"),
            # ... (contenu de la page Interprétation locale)
        ])
    elif tab_name == 'global_interpretation':
        # ... (contenu de la page Interprétation globale)
        return html.Div([
            html.H2("Interprétation globale"),
            # ... (contenu de la page Interprétation globale)
        ])

# Callback pour mettre à jour le résultat de la demande
@app.callback(
    [Output('prediction-output', 'children'),
     Output('client-info-expander-output', 'children')],
    [Input('start-button', 'n_clicks')],
    [State('client-dropdown', 'value')]
)
def update_prediction_output(n_clicks, client_id):
    if n_clicks is None:
        return '', ''
    else:
        if client_id and client_id != '<Select>':
            probability, decision = get_prediction(client_id)
            result_output = ''
            client_info_output = ''
            if decision == 'Accordé':
                result_output = html.Div([
                    html.H2("RÉSULTAT DE LA DEMANDE"),
                    html.Div("Crédit accordé", style={'color': 'green'}),
                    dcc.Graph(figure=plot_score_gauge(probability))
                ])
            else:
                result_output = html.Div([
                    html.H2("RÉSULTAT DE LA DEMANDE"),
                    html.Div("Crédit refusé", style={'color': 'red'})
                ])

            if client_id:
                client_info = pd.DataFrame(data_test.loc[data_test.index == client_id])
                client_info_output = html.Div([
                    html.Div("Voici les informations du client:"),
                    dcc.Markdown(client_info.to_markdown())
                ])

            return result_output, client_info_output
        else:
            return '', ''


# Point d'entrée de l'application Dash
if __name__ == '__main__':
    app.run_server(debug=True, port=9000)
