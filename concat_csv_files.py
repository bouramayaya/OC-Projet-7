import pandas as pd
import requests
from io import StringIO

# URL brute du fichier CSV sur GitHub
github_csv_url = 'https://github.com/bouramayaya/dashboard/blob/master/data/X_train_1.csv'

# Utilisez requests pour obtenir le contenu du fichier CSV
response = requests.get(github_csv_url)

# Vérifiez si la requête a réussi (statut 200)
if response.status_code == 200:
    # Utilisez StringIO pour lire le contenu comme un fichier
    csv_content = StringIO(response.text)
    
    # Utilisez pandas pour lire le CSV depuis le contenu
    df = pd.read_csv(csv_content)

    # Affichez les premières lignes du DataFrame
    print(df.head())
else:
    print(f"Échec de la requête avec le code d'état {response.status_code}")
