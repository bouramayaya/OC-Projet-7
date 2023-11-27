import os
import pandas as pd

def concat_csv_files(folder_path, keyword):
    # Obtenez la liste des fichiers dans le dossier
    files = [file for file in os.listdir(folder_path) if file.endswith('.csv') and keyword in file]

    # Vérifiez si des fichiers correspondants ont été trouvés
    if not files:
        print(f"Aucun fichier correspondant au mot-clé '{keyword}' trouvé.")
        return None

    # Initialisez une liste pour stocker les DataFrames
    dfs = []

    # Parcourez chaque fichier, lisez-le et ajoutez le DataFrame à la liste
    for file in files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        dfs.append(df)

    # Concaténez tous les DataFrames dans la liste en un seul DataFrame
    concatenated_df = pd.concat(dfs, ignore_index=True)

    return concatenated_df


folder_path = '/home/ubuntu/OC/OC-Projet-7/data'
folder_path = 'C:/Users/Fane0763/OpenClassroom/OC Projet 7/OC-Projet-7/data'
keyword = 'X_train_'
result_df = concat_csv_files(folder_path, keyword)
print(result_df.shape)
