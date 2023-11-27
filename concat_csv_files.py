
import pandas as pd
import numpy as np
import requests
from io import StringIO

# ------------------------------------------------------------------------------------------------------------
# Chargement des données
# ------------------------------------------------------------------------------------------------------------
def charger_et_concatener_fichiers_github(nom_utilisateur, nom_repo, chemin_dossier, mot_cle):
    url = f"https://api.github.com/repos/{nom_utilisateur}/{nom_repo}/contents/{chemin_dossier}"
    response = requests.get(url)
    fichiers_csv = []

    if response.status_code == 200:
        fichiers = response.json()
        for fichier in fichiers:
            if fichier["name"].lower().endswith('.csv') and mot_cle in fichier["name"]:
                contenu = requests.get(fichier["download_url"]).text
                dataframe = pd.read_csv(StringIO(contenu))
                fichiers_csv.append(dataframe)

        if not fichiers_csv:
            print(f"Aucun fichier contenant le mot-clé '{mot_cle}' n'a été trouvé dans le dossier GitHub.")
            return None

        dataframe_concatene = pd.concat(fichiers_csv, axis=0, ignore_index=True)
        return dataframe_concatene.set_index('SK_ID_CURR')
    else:
        print(f"Erreur lors de la récupération des fichiers : {response.status_code}")
        return None


nom_utilisateur = "bouramayaya"
nom_repo = "OC-Projet-7"
chemin_dossier = "data"

data_test = charger_et_concatener_fichiers_github(nom_utilisateur, nom_repo, chemin_dossier, 'test_df')
data_train = charger_et_concatener_fichiers_github(nom_utilisateur, nom_repo, chemin_dossier, 'train_df_1')
X_train = charger_et_concatener_fichiers_github(nom_utilisateur, nom_repo, chemin_dossier, 'X_train_1')

print(data_test.shape)
