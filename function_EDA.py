import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

# Plotly
# import plotlypython.exe -m pip install --upgrade pip
import plotly.graph_objects as go
from plotly.subplots import make_subplots

font_title = {'family': 'serif', 'color': '#114b98', 'weight': 'bold', 'size': 16, }
font_title2 = {'family': 'serif', 'color': '#114b98', 'weight': 'bold', 'size': 12, }
font_title3 = {'family': 'serif', 'color': '#4F6272', 'weight': 'bold', 'size': 10, }

mycolors = ["black", "hotpink", "b", "#4CAF50"]
AllColors = ['#99ff99', '#66b3ff', '#4F6272', '#B7C3F3', '#ff9999', '#ffcc99', '#ff6666', '#DD7596', '#8EB897',
             '#c2c2f0', '#DDA0DD', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
             '#7f7f7f', '#bcbd22', '#17becf', '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33',
             '#a65628', '#f781bf', "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
             "#000000"]
# date
from datetime import time


class clr:
    start = '\033[93m' + '\033[1m'
    color = '\033[93m'
    end = '\033[0m'


random_state = 42
SAVE_IMAGES = True
# -------------------------------------------------------------------------------------------
#  Reglages
# --------------------------------------------------------------------------------------------

import sys


def is_colab_environment():
    # Vérifier si le module 'google.colab' est présent dans la liste des modules importés
    return 'google.colab' in sys.modules


# Exemple d'utilisation
if is_colab_environment():
    print("Le code s'exécute dans l'environnement Google Colab.")
else:
    print("Le code s'exécute dans un environnement local.")


# -------------------------------------------------------------------------------------------
#  Fonctions qui gèrent les chemins
# --------------------------------------------------------------------------------------------


def os_make_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def os_path_join(folder, file):
    """
    Remplacement pour `os.path.join(folder, file)` sur Windows
    """
    return f'{folder}/{file}'


# Configuration des chemins de dossiers pour les différents environnements

if not is_colab_environment():
    PROJ_FOLDER = 'C:/Users/Fane0763/OpenClassroom/OC Projet 7'
    DATA_FOLDER = os_path_join(f'{PROJ_FOLDER}', 'bases')
    OUT_FOLDER = os_path_join(f'{PROJ_FOLDER}', 'out_put')
    GRAPH_FOLDER = os_path_join(f'{PROJ_FOLDER}', 'Graphs')  # graphique pour les diapos

# Définition des chemins de dossiers pour l'environnement Colab
if is_colab_environment():
    # Colaboratory - décommentez les 2 lignes suivantes pour connecter à votre drive
    # from google.colab import drive
    # drive.mount('/content/drive')
    PROJ_FOLDER = '/content/drive/MyDrive/OC-Projet-7'
    DATA_FOLDER = os_path_join(f'{PROJ_FOLDER}', 'bases')
    OUT_FOLDER = os_path_join(f'{PROJ_FOLDER}', 'out_put')
    GRAPH_FOLDER = os_path_join(f'{PROJ_FOLDER}', 'Graphs')  # graphique pour les diapos


# imgPath = f'{GRAPH_FOLDER}/'

# -------------------------------------------------------------------------------------------
# Sauvegarder les graphiques
# -------------------------------------------------------------------------------------------

def fig_name_cleaning(fig_name: str) -> str:
    """Enlever les caractères interdits dans les filenames ou filepaths"""
    return fig_name.replace(' ', '_').replace(':', '-').replace(
        '.', '-').replace('/', '_').replace('>', 'gt.').replace('<', 'lt.')


def to_png(fig_name=None) -> None:
    """
    Register the current plot figure as an image in a file.
    Must call plt.show() or show image (by calling to_png() as last row in python cell)
    to apply the call 'bbox_inches=tight', to be sure to include the whole title / legend
    in the plot area.
    """

    def get_title() -> str:
        """find current plot title (or suptitle if more than one plot)"""
        if plt.gcf()._suptitle is None:  # noqa
            return plt.gca().get_title()
        else:
            return plt.gcf()._suptitle.get_text()  # noqa

    if SAVE_IMAGES:
        if fig_name is None:
            fig_name = get_title()
        elif len(fig_name) < 9:
            fig_name = f'{fig_name}_{get_title()}'
        fig_name = fig_name_cleaning(fig_name)
        print(f'"{fig_name}.png"')
        plt.gcf().savefig(
            # f'{IMAGE_FOLDER}/{fig_name}.png', bbox_inches='tight')
            os_path_join(f'{GRAPH_FOLDER}', f'{fig_name}.png'), bbox_inches='tight')


# -------------------------------------------------------------------------------------------
# Nom d'une base en string
# -------------------------------------------------------------------------------------------

def get_dataframe_name(df):
    name = [x for x in globals() if globals()[x] is df]
    return name[0]


def namestr(df):
    name = [name for name in globals() if globals()[name] is df]
    return name[0] if name else None


# -------------------------------------------------------------------------------------------
# Aperçu de la base de données pandas dataframe
# -------------------------------------------------------------------------------------------

def apercu(datasets, titles):
    # Importations
    # import pandas as pd

    # datasets = [customers_df, geolocation_df, items_df, payments_df, reviews_df, orders_df, products_df, sellers_df,
    # category_translation_df]
    # titles=[namestr(data,globals()) for data in datasets]
    data_summary = pd.DataFrame({}, )
    data_summary['datasets'] = titles
    data_summary['columns'] = [', '.join([col for col in data.columns]) for data in datasets]
    data_summary['nb_lignes'] = [data.shape[0] for data in datasets]
    data_summary['nb_colonnes'] = [data.shape[1] for data in datasets]
    data_summary['doublons'] = [data.duplicated().sum() for data in datasets]
    data_summary['nb_NaN'] = [data.isnull().sum().sum() for data in datasets]
    data_summary['NaN_Columns'] = [', '.join([col for col, null in data.isnull().sum().items() if null > 0]) for data in
                                   datasets]
    return data_summary.style.background_gradient(cmap='YlGnBu')


def apercu2(datasets, titles):
    data_summary = pd.DataFrame({}, )
    data_summary['datasets'] = titles
    data_summary['nb_lignes'] = [data.shape[0] for data in datasets]
    data_summary['nb_colonnes'] = [data.shape[1] for data in datasets]
    data_summary['doublons'] = [data.duplicated().sum() for data in datasets]
    data_summary['nb_NaN'] = [data.isnull().sum().sum() for data in datasets]

    return data_summary.style.background_gradient(cmap='YlGnBu')


def infoDataFrame(data):
    """
    Cette fonction affiche les stats relatives a la dataframe en entrée
    :param data: dataframe pandas
    :return: dataframe avec nombre de lignes et de colonnes et des infos sur le dataframe
    """
    display(Markdown('------------------------------------'))
    display(Markdown('#### Info générales sur la base : {0}'.format(namestr(data))))
    display(Markdown('------------------------------------'))
    # print('--------------------------------------------------------------------------')
    # print('Info générales sur la base : {0}'.format(namestr(data, globals())))
    data.info()
    print(" ")
    print(" ")
    nb_ligne = data.shape[0]
    nb_colonne = data.shape[1]
    print('Le jeu de données {} a {} lignes et {} colonnes.'.format(namestr(data),
                                                                    nb_ligne, nb_colonne))
    df = pd.DataFrame({'Variable': ['lignes', 'colonnes'], 'nombre': [nb_ligne, nb_colonne]})
    print(" ")
    # display(df)
    return df


def Explodetuple(m):
    liste1 = []
    for t in range(m):
        if t in [0, 1]:
            liste1.append(0.1)
        else:
            liste1.append(0)
    return tuple(liste1)


def percentFreq(values):
    def my_format(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        # return '{:.1f}%\n({v:d})'.format(pct, v=val)
        return '{:.1f}%({v:d})'.format(pct, v=val)

    return my_format


def repartitionTypeVar(data, figsize=(6, 3),
                       title="Repartition par types de variables \n",
                       graphName=None):
    df = data.dtypes.value_counts()
    L = len(df)
    labels = list(df.index)
    sizes = list(df)
    # print(labels,"\n",sizes)
    explode = Explodetuple(L)
    colors = AllColors[:L]
    fig1, ax1 = plt.subplots(figsize=figsize)
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct=percentFreq(df), shadow=True, startangle=0)
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')
    plt.tight_layout()
    plt.title(label=title, fontdict=font_title)
    plt.legend()
    if graphName:
        plt.savefig(os_path_join(f'{GRAPH_FOLDER}', graphName),
                    bbox_inches='tight')
    plt.show()
    plt.close()
    df = df.reset_index()
    df.columns = ['Types de variables', 'Nombre']
    display(df.reset_index(drop=True))


# -------------------------------------------------------------------------------------------
# Taux de remplissage d'un dataframe
# -------------------------------------------------------------------------------------------
def fillingRate(data: pd.DataFrame, GRAPH_FOLDER, grahName: str = None):
    filled = data.notna().sum().sum() / (data.shape[0] * data.shape[1])
    missing = data.isna().sum().sum() / (data.shape[0] * data.shape[1])
    taux = [filled, missing]
    labels = ["%filled", "%missing"]
    explode = (0.1, 0)
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.title("Taux de completion \n", fontdict=font_title)
    ax.axis("equal")
    ax.pie(taux, explode=explode, labels=labels, autopct='%1.2f%%', shadow=True, )
    plt.legend(labels)
    if grahName:
        plt.savefig(os_path_join(GRAPH_FOLDER, grahName), bbox_inches='tight')
    plt.show()
    plt.close()


# -------------------------------------------------------------------------------------------
# Recherche de liste de variables a travers des suffixes
# -------------------------------------------------------------------------------------------

def searchfeature(data, suffix):
    varList = [col for col in data.columns if suffix in col]
    return varList


# -------------------------------------------------------------------------------------------
# Liste des  fichiers et tailles
# -------------------------------------------------------------------------------------------
def get_size_str(octets):
    """
    Get size of file in octets
    """
    g_b = round(octets / 2 ** 30, 2)
    m_b = round(octets / 2 ** 20, 2)
    k_b = round(octets / 2 ** 10, 2)
    if g_b > 1:
        ret = f'{g_b} Go'
    elif m_b > 1:
        ret = f'{m_b} Mo'
    elif k_b > 1:
        ret = f'{k_b} ko'
    else:
        ret = f'{octets} octets'
    return ret


def list_files_and_sizes(directory_path):
    file_list = []
    size_list = []

    for filename in os.listdir(directory_path):
        file_path = os_path_join(directory_path, filename)
        if os.path.isfile(file_path):
            file_list.append(filename)
            size_list.append(get_size_str(os.path.getsize(file_path)))

    data = {'File Name': file_list, 'File Size': size_list}
    df = pd.DataFrame(data)
    return df


# --------------------------------------------------------------------
# -- CREATION DATAFRAME DES VALEURS MANQUANTES
# --------------------------------------------------------------------


def nan_df_create(data):
    '''
    Function to create a dataframe of percentage of NaN values for each column of the dataframe
    Inputs:
        data:
            DataFrame

    Returns:
        DataFrame of NaN percentages
    '''
    nan_percentages = data.isna().sum() * 100 / len(data)
    df = pd.DataFrame({'column': nan_percentages.index,
                       'percent': nan_percentages.values})

    # sorting the dataframe by decreasing order of percentage of NaN values
    df.sort_values(by='percent', ascending=False, inplace=True)

    return df


# --------------------------------------------------------------------
# -- REPRESENTATION BARPLOT DES VALEURS MANQUANTES PAR VARIABLE
# --------------------------------------------------------------------


def plot_nan_percent(df_nan, title_name, tight_layout=True, figsize=(20, 8),
                     grid=False, rotation=90, fontsize=12):
    '''
    Function to plot Bar Plots of NaN percentages for each Column with missing values

    Inputs:
        df_nan:
            DataFrame of NaN percentages
        title_name:
            Name of table to be displayed in title of plot
        tight_layout: bool, default = True
            Whether to keep tight layout or not
        figsize: tuple, default = (20,8)
            Figure size of plot
        grid: bool, default = False
            Whether to draw gridlines to plot or not
        rotation: int, default = 0
            Degree of rotation for x-tick labels
    '''
    # sns.set(style='white', font_scale=1.2)
    # checking if there is any column with NaNs or not.
    if df_nan.percent.sum() != 0:
        print(
            f"Nombre de variables avec valeurs manquantes : {df_nan[df_nan['percent'] != 0].shape[0]}")

        # plotting the Bar-Plot for NaN percentages (only for columns with
        # Non-Zero percentage of NaN values)
        plt.figure(figsize=figsize, tight_layout=tight_layout)
        sns.barplot(x='column', y='percent',
                    data=df_nan[df_nan['percent'] > 0])
        plt.xticks(rotation=rotation)
        plt.xlabel('Nom de variable', fontsize=fontsize)
        plt.ylabel('% de valeurs NaN', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.title(f'Pourcentage de valeurs NaN dans {title_name}', fontsize=fontsize + 4)
        if grid:
            plt.grid()
        plt.show()
    else:
        print(
            f"Le dataframe {title_name} ne contient pas de valeurs manquantes.")


def data_duplicated(df):
    '''
    Retourne le nombres de lignes identiques.
    '''
    return df.duplicated(keep=False).sum()


def row_duplicated(df, col):
    '''
    Retourne le nombre de doublons de la variables col.
    '''
    return df.duplicated(subset=col, keep='first').sum()


def missing_cells(df):
    '''
    Calcule le nombre de cellules manquantes sur le data set total
    '''
    return df.isna().sum().sum()


def missing_cells_perc(df):
    '''Calcule le pourcentage de cellules manquantes sur le data set total'''
    return df.isna().sum().sum() / (df.size)


def missing_general(df):
    '''
    Donne un aperçu général du nombre de données manquantes dans le data frame
    '''
    print('Nombre total de cellules manquantes :', missing_cells(df))
    print('Nombre de cellules manquantes en %  : {:.2%}'.format(missing_cells_perc(df)))


def valeurs_manquantes(df):
    '''Prend un data frame en entrée et créer en sortie un dataframe contenant le nombre de valeurs manquantes
    et leur pourcentage pour chaque variables. '''
    tab_missing = pd.DataFrame(columns=['Variable', 'Missing values', 'Missing (%)'])
    tab_missing['Variable'] = df.columns
    missing_val = list()
    missing_perc = list()

    for var in df.columns:
        nb_miss = missing_cells(df[var])
        missing_val.append(nb_miss)
        perc_miss = missing_cells_perc(df[var])
        missing_perc.append(perc_miss)

    tab_missing['Missing values'] = list(missing_val)
    tab_missing['Missing (%)'] = list(missing_perc)
    return tab_missing


def drop_columns_empty(df, lim):
    '''Prend en entrée un data frame et un seuil de remplissage de données.
    Supprime chaque variable ayant un pourcentage de données manquantes supérieur à celui renseigné.
    Donne en sortie le data frame filtré avec les colonnes à garder.'''

    tab = valeurs_manquantes(df)
    columns_keep = list()
    for row in tab.iterrows():
        if float(row[1]['Missing (%)']) > float(lim):
            print('Suppression de la variable {} avec % de valeurs manquantes {}'.format(row[1]['Variable'],
                                                                                         round(float(
                                                                                             row[1]['Missing (%)']),
                                                                                             2)))

        else:
            columns_keep.append(row[1]['Variable'])

    return df[columns_keep]


def exploration_missings(data):
    print('Types de variables de :', namestr(data))
    # Affichage des types de variables dans la base.
    repartitionTypeVar(data, figsize=(7, 4),
                       graphName=f'RepartitionTypeVar_{namestr(data)}.png')
    print(clr.color + '*' * 110 + clr.end)
    print('Taux de complétion de :', namestr(data))
    print(clr.color + '*' * 110 + clr.end)
    fillingRate(data, f'{GRAPH_FOLDER}',
                f'tauxCompletion_{namestr(data)}.png')
    print(clr.color + '*' * 110 + clr.end)
    print('Heatmap des valeurs manquantes :', namestr(data))
    print(clr.color + '*' * 110 + clr.end)
    # msno.matrix(data)
    plt.figure(figsize=(15, 7))
    sns.heatmap(data.isna(), cbar=False)
    plt.show()

    print(clr.color + '*' * 110 + clr.end)
    print('Réprésentation visuelle des valeurs manquantes :', namestr(data))
    print(clr.color + '*' * 110 + clr.end)

    df_nan = nan_df_create(data)
    plot_nan_percent(df_nan, namestr(data), tight_layout=True, figsize=(50, 30),
                     grid=False, rotation=90, fontsize=12)


def exploration_dataframes(data, df_cols_desc, str_match=None):
    print('Description des variables de :', namestr(data))
    if str_match:
        df_var = df_cols_desc[df_cols_desc['Table'].str.match(str_match)]
        display(df_var)

    print('Types de variables de :', namestr(data))
    # Affichage des types de variables dans la base.
    repartitionTypeVar(data, figsize=(7, 4),
                       graphName=f'RepartitionTypeVar_{namestr(data)}.png')
    print(clr.color + '*' * 120 + clr.end)
    print('Taux de complétion de :', namestr(data))
    print(clr.color + '*' * 120 + clr.end)
    fillingRate(data, f'{GRAPH_FOLDER}',
                f'tauxCompletion_{namestr(data)}.png')
    print(clr.color + '*' * 120 + clr.end)
    print('Heatmap des valeurs manquantes :', namestr(data))
    print(clr.color + '*' * 120 + clr.end)
    # msno.matrix(data)
    sns.heatmap(data.isna(), cbar=False)

    print(clr.color + '*' * 120 + clr.end)
    print('Réprésentation visuelle des valeurs manquantes :', namestr(data))
    print(clr.color + '*' * 120 + clr.end)

    df_nan = nan_df_create(data)
    plot_nan_percent(df_nan, namestr(data), tight_layout=True, figsize=(20, 12),
                     grid=False, rotation=90, fontsize=12)


def impute_with_interpolate(dataframe):
    # Crée une copie du DataFrame d'origine pour éviter de modifier les données d'origine.
    short_cleaned_impute = dataframe.copy()
    # Parcours chaque colonne du DataFrame.
    for col_name in dataframe:
        # Utilise la méthode d'interpolation linéaire pour remplir les valeurs manquantes.
        short_cleaned_impute[col_name] = dataframe[col_name].interpolate(
            method='linear', inplace=False, limit_direction="both"
        ).ffill().bfill()
    return short_cleaned_impute


from sklearn.impute import SimpleImputer


def impute_with_interpolate2(data: pd.DataFrame):
    dataframe = data.copy()
    numeric_cols = dataframe.select_dtypes(include=np.number).columns
    categorical_cols = dataframe.select_dtypes(exclude=np.number).columns
    imputer_categorical = SimpleImputer(strategy='most_frequent')
    dataframe[categorical_cols] = imputer_categorical.fit_transform(dataframe[categorical_cols])
    dataframe[numeric_cols] = dataframe[numeric_cols].interpolate(
        method='linear', inplace=False, limit_direction="both").ffill().bfill()
    return dataframe


def colonnes_numeriques(dataframe):
    """
    Récupère la liste des colonnes numériques d'un DataFrame.
    """
    colonnes_num = dataframe.select_dtypes(include=['number']).columns.tolist()
    return colonnes_num


def colonnes_cat(dataframe):
    """
    Récupère la liste des colonnes numériques d'un DataFrame.
    """
    colonnes_cat = dataframe.select_dtypes(include=['object']).columns.tolist()
    return colonnes_cat


from sklearn.impute import KNNImputer


def impute_with_knn(data: pd.DataFrame, cols_list: list, n_neighbors=5):
    df = data.copy()  # Une copie du dataframe
    numeric_cols = df.select_dtypes(include=np.number).columns
    imputer = KNNImputer(n_neighbors=n_neighbors)  # instance de l'imputeur KNN
    df_imputed = imputer.fit_transform(df[numeric_cols])  # imputation KNN sur le DataFrame
    df_imputed = pd.DataFrame(df_imputed, columns=df.columns)  # Reconvertit en DataFrame
    return df_imputed


# --------------------------------------------------------------------
# -- AFFICHE LA LISTE DES IDENTIFIANTS UNIQUES
# --------------------------------------------------------------------

def print_unique_categories(data, column_name, show_counts=False):
    '''
    Function to print the basic stats such as unique categories and their counts for categorical variables

        Inputs:
        data: DataFrame
            The DataFrame from which to print statistics
        column_name: str
            Column's name whose stats are to be printed
        show_counts: bool, default = False
            Whether to show counts of each category or not

    '''

    print('-' * 79)
    print(
        f"Les catégories uniques de la variable '{column_name}' sont :\n{data[column_name].unique()}")
    print('-' * 79)

    if show_counts:
        print(
            f"Répartition dans chaque catégorie :\n{data[column_name].value_counts()}")
        print('-' * 79)


# --------------------------------------------------------------------
# -- BARPLOT DES VARIABLES CATEGORIELLES
# --------------------------------------------------------------------


def plot_categorical_variables_bar(data, column_name, figsize=(18, 6),
                                   percentage_display=True,
                                   plot_defaulter=True, rotation=0,
                                   horizontal_adjust=0,
                                   fontsize_percent='xx-small',
                                   palette1='Set1',
                                   palette2='Set2'):
    '''
    Function to plot Categorical Variables Bar Plots

    Inputs:
        data: DataFrame
            The DataFrame from which to plot
        column_name: str
            Column's name whose distribution is to be plotted
        figsize: tuple, default = (18,6)
            Size of the figure to be plotted
        percentage_display: bool, default = True
            Whether to display the percentages on top of Bars in Bar-Plot
        plot_defaulter: bool
            Whether to plot the Bar Plots for Defaulters or not
        rotation: int, default = 0
            Degree of rotation for x-tick labels
        horizontal_adjust: int, default = 0
            Horizontal adjustment parameter for percentages displayed on the top of Bars of Bar-Plot
        fontsize_percent: str, default = 'xx-small'
            Fontsize for percentage Display

    '''

    print(
        f"Nombre de catégories uniques pour {column_name} = {len(data[column_name].unique())}")

    plt.figure(figsize=figsize, tight_layout=True)
    sns.set(style='whitegrid', font_scale=1.2)

    # plotting overall distribution of category
    plt.subplot(1, 2, 1)
    data_to_plot = data[column_name].value_counts().sort_values(ascending=False)
    ax = sns.barplot(x=data_to_plot.index, y=data_to_plot, palette=palette1)

    if percentage_display:
        total_datapoints = len(data[column_name].dropna())
        for p in ax.patches:
            ax.text(
                p.get_x() +
                horizontal_adjust,
                p.get_height() +
                0.005 *
                total_datapoints,
                '{:1.02f}%'.format(
                    p.get_height() *
                    100 /
                    total_datapoints),
                fontsize=fontsize_percent)

    plt.xlabel(column_name, labelpad=10)
    plt.title('Toutes TARGET', pad=20, fontsize=30)
    plt.xticks(rotation=rotation, fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel('Nombre', fontsize=20)

    # plotting distribution of category for Defaulters
    if plot_defaulter:
        percentage_defaulter_per_category = (data[column_name][data.TARGET == 1].value_counts(
        ) * 100 / data[column_name].value_counts()).dropna().sort_values(ascending=False)

        plt.subplot(1, 2, 2)
        sns.barplot(x=percentage_defaulter_per_category.index,
                    y=percentage_defaulter_per_category, palette=palette2)
        plt.ylabel(
            'Pourcentage par catégorie pour les défaillants',
            fontsize=20)
        plt.xlabel(column_name, labelpad=10)
        plt.xticks(rotation=rotation, fontsize=20)
        plt.yticks(fontsize=20)
        plt.title('Défaillants seuls', pad=20, fontsize=30)

    plt.suptitle(f'Répartition de {column_name}', fontsize=40)
    plt.show()


def plot_categorical_variable_bar(data, column_name, figsize=(18, 6),
                                  percentage_display=True, rotation=0,
                                  horizontal_adjust=0,
                                  fontsize_percent='xx-small',
                                  palette1='Set1'):
    '''
    Function to plot Categorical Variables Bar Plots

    Inputs:
        data: DataFrame
            The DataFrame from which to plot
        column_name: str
            Column's name whose distribution is to be plotted
        figsize: tuple, default = (18,6)
            Size of the figure to be plotted
        percentage_display: bool, default = True
            Whether to display the percentages on top of Bars in Bar-Plot
        plot_defaulter: bool
            Whether to plot the Bar Plots for Defaulters or not
        rotation: int, default = 0
            Degree of rotation for x-tick labels
        horizontal_adjust: int, default = 0
            Horizontal adjustment parameter for percentages displayed on the top of Bars of Bar-Plot
        fontsize_percent: str, default = 'xx-small'
            Fontsize for percentage Display

    '''

    print(
        f"Nombre de catégories uniques pour {column_name} = {len(data[column_name].unique())}")

    plt.figure(figsize=figsize, tight_layout=True)
    sns.set(style='whitegrid', font_scale=1.2)

    data_to_plot = data[column_name].value_counts().sort_values(ascending=False)
    ax = sns.barplot(x=data_to_plot.index, y=data_to_plot, palette=palette1)

    if percentage_display:
        total_datapoints = len(data[column_name].dropna())
        for p in ax.patches:
            ax.text(
                p.get_x() +
                horizontal_adjust,
                p.get_height() +
                0.005 *
                total_datapoints,
                '{:1.02f}%'.format(
                    p.get_height() *
                    100 /
                    total_datapoints),
                fontsize=fontsize_percent)

    plt.xlabel(column_name, labelpad=10)
    plt.title(f'Barplot de {column_name}', pad=20, fontsize=30)
    plt.xticks(rotation=rotation, fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel('Nombre', fontsize=20)

    plt.show()


# --------------------------------------------------------------------
# -- PIEPLOT DES VARIABLES CATEGORIELLES
# --------------------------------------------------------------------


def plot_categorical_variables_pie(
        data,
        column_name,
        plot_defaulter=True,
        hole=0):
    '''
    Function to plot categorical variables Pie Plots
    Inputs:
        data: DataFrame
            The DataFrame from which to plot
        column_name: str
            Column's name whose distribution is to be plotted
        plot_defaulter: bool
            Whether to plot the Pie Plot for Defaulters or not
        hole: int, default = 0
            Radius of hole to be cut out from Pie Chart
    '''
    if plot_defaulter:
        cols = 2
        specs = [[{'type': 'domain'}, {'type': 'domain'}]]
        titles = ['Toutes TARGET', 'Défaillants seuls']
    else:
        cols = 1
        specs = [[{'type': 'domain'}]]
        titles = [f'Répartition de la variable {column_name}']

    values_categorical = data[column_name].value_counts()
    labels_categorical = values_categorical.index

    fig = make_subplots(rows=1, cols=cols,
                        specs=specs,
                        subplot_titles=titles)

    fig.add_trace(
        go.Pie(
            values=values_categorical,
            labels=labels_categorical,
            hole=hole,
            textinfo='percent',
            textposition='inside'),
        row=1,
        col=1)

    if plot_defaulter:
        percentage_defaulter_per_category = data[column_name][data.TARGET == 1].value_counts(
        ) * 100 / data[column_name].value_counts()
        percentage_defaulter_per_category.dropna(inplace=True)
        percentage_defaulter_per_category = percentage_defaulter_per_category.round(
            2)

        fig.add_trace(
            go.Pie(
                values=percentage_defaulter_per_category,
                labels=percentage_defaulter_per_category.index,
                hole=hole,
                textinfo='percent',
                hoverinfo='label+value'),
            row=1,
            col=2)

    fig.update_layout(title=f'Répartition de la variable {column_name}')
    fig.show()


# ----------------------------------------------------
#     Normalisation des données
# ----------------------------------------------------

from sklearn import preprocessing


def scale_dataframe(df: pd.DataFrame, scaler=None) -> pd.DataFrame:
    if scaler is None:
        scaler = preprocessing.StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)


from termcolor import colored
from sklearn.metrics import fbeta_score as calcul_fbeta_score
import time


def classification(best_param, X_train, y_train, X_test, y_test, algo_name, file_name=None):
    print(colored("Exécution en cours ....\n", 'blue'))

    start_time = time.time()
    model = best_param.fit(X_train, y_train)
    end_time = time.time()
    execution_time = end_time - start_time

    if file_name is None:
        pass
    else:
        # save the model to disk
        filename = file_name
        pickle.dump(model, open(filename, 'wb'))

    start_time = time.time()
    prediction = best_param.predict(X_test)
    end_time = time.time()
    prediction_time = end_time - start_time

    # prediction de probabilité d'appartenance à 0 et 1
    probability = best_param.predict_proba(X_test)
    probability_positive = probability[:, 1]
    resume_prediction_class = pd.DataFrame(
        {'cat_reel': y_test, 'cat_predit': prediction}).reset_index()

    probability_positive_class = pd.DataFrame(
        {'SK_ID_CURR': X_test.index, 'positive_probability': probability_positive})

    # print(probability_positive_class)

    print(f"Qualité de la classification {algo_name} \n ")
    accuracy = accuracy_score(y_test, prediction)
    print("L'accuracy score est de      : {}".format(round(accuracy, 5)))
    precision = precision_score(y_test, prediction,
                                average='binary')  # binary car ici 0 ou 1 si plusieurs label c'est weighted
    print('Le score de précision est de : {}'.format(round(precision, 5)))
    recall = recall_score(y_test, prediction, average='binary')
    print('Le score recall est de       : {}'.format(round(recall, 5)))

    f1 = f1_score(y_test, prediction, average='binary')
    print('Le score f1 est de           : {}'.format(round(f1, 5)))
    AUC = roc_auc_score(y_test, probability[:, 1])  # proba de na pas rembourser prêt donc d'être 1
    print('Le score AUC est de          : {}'.format(round(AUC, 5)))
    fbeta_score = calcul_fbeta_score(y_test, prediction, beta=2)

    return model, accuracy, precision, recall, f1, AUC, fbeta_score, prediction, probability, \
        execution_time, prediction_time, resume_prediction_class, probability_positive_class


def matrix_TN_FN(y_test, y_prediction):
    """
    Cette fonction retourne une matrice de confusion.
    """

    # Création de la matrice de confusion.
    df_matrice_confusion = pd.DataFrame(columns=['Predicted Negative (0)', 'Predicted Positive (1)'],
                                        index=['Real Negative (0)', 'Real Positive (1)'])

    # DataFrame de comparaison.
    df_pred_compare = pd.concat([pd.Series(y_test.reset_index(drop=True)), pd.Series(y_prediction)], axis=1)
    df_pred_compare.columns = ['Real category', 'Prediction']

    # Masque suivant les tp,tn, fp...
    mask_real_pos = (df_pred_compare['Real category'] == 1)
    mask_pred_pos = (df_pred_compare['Prediction'] == 1)

    mask_real_neg = (df_pred_compare['Real category'] == 0)
    mask_pred_neg = (df_pred_compare['Prediction'] == 0)

    # Négatif.
    true_negative = df_pred_compare[mask_real_neg & mask_pred_neg].shape[0]
    false_negative = df_pred_compare[mask_real_pos & mask_pred_neg].shape[0]

    # Positif.
    false_positive = df_pred_compare[mask_real_neg & mask_pred_pos].shape[0]
    true_positive = df_pred_compare[mask_real_pos & mask_pred_pos].shape[0]

    # Remplissage de la matrice.
    df_matrice_confusion['Predicted Negative (0)'] = ["{} (TN)".format(true_negative), "{} (FN)".format(false_negative)]
    df_matrice_confusion['Predicted Positive (1)'] = ["{} (FP)".format(false_positive), "{} (TP)".format(true_positive)]

    return df_matrice_confusion


from sklearn.metrics import roc_curve, roc_auc_score


def cf_matrix_roc_auc(y_true, y_pred, y_pred_proba, size):
    """
    This function will make a pretty plot of
     an sklearn Confusion Matrix using a Seaborn heatmap visualization + ROC Curve.
     """

    fig = plt.figure(figsize=size)
    ax1 = fig.add_subplot(221)
    ax1.title.set_text('Confusion Matrix')

    cf_matrix = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    group_names = ['True Neg (TN)', 'False Pos (FP)', 'False Neg (FN)', 'True Pos (TP)']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='RdPu')
    plt.tick_params(axis='both', which='major', labelsize=10, labelbottom=False, bottom=False,
                    top=False, labeltop=True)

    AUC = roc_auc_score(y_true, y_pred_proba[:, 1])
    # plt.subplot(222)
    ax2 = fig.add_subplot(222)
    ax2.title.set_text('ROC Curve')

    fpr, tpr, thresholds = roc_curve(y_true,
                                     y_pred_proba[:, 1])  # pour la courbe ROC utilisation de la probabiilité d'être 1
    plt.plot(fpr, tpr, color='pink', linewidth=5, label='AUC = {:.4f}'.format(AUC))
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

    print('True negative   = ', tn)
    print('False positive  = ', fp)
    print('False negative  = ', fn)
    print('True positive   = ', tp)

    return tn, fp, fn, tp


def fonction_metier(y_true, y_pred):
    '''Créer un score métier à partir de la matrice de confusion.
    :param: y_true (vraies valeurs), y_pred (valeurs prédites par le modèle)
    :return: gain (score métier)
    '''
    TP_coeff = 0  # Vrais positifs
    FP_coeff = -1  # Faux positifs (prédit comme faisant défaut (1) mais ne fait pas défaut (0))
    FN_coeff = -100  # Faux négatifs (prédit comme ne faisant pas défaut (0) mais font défaut (1))
    TN_coeff = 10  # Vrais négatifs

    (TN, FP, FN, TP) = confusion_matrix(y_true, y_pred).ravel()

    gain = (TP * TP_coeff + TN * TN_coeff + FP * FP_coeff + FN * FN_coeff) / (TN + FP + FN + TP)

    return gain


def score_banq(tn, fp, fn, tp, coeff_tn, coeff_fp, coeff_fn, coeff_tp):
    total = (coeff_tn * tn + coeff_fp * fp + coeff_fn * fn + coeff_tp * tp)
    # calcul du gain maximal que peut avoir la banque quand il n'y a pas de perte d'argent. C'est à dire que le modèle ne détecte que TN et les TP.
    max_gain = (tn + fp) * coeff_tn + (tp + fn) * coeff_tp
    # calcul du gain minimal (perte) que peut avoir la banque quand il n'y a pas de gain d'argent. C'est à dire que le modèle ne détecte que FN et les FP.
    min_gain = (tn + fp) * coeff_fp + (tp + fn) * coeff_fn
    # normalisation min-max feature scalling : score= (tot-min)/(max-min) https://en.wikipedia.org/wiki/Normalization_(statistics)
    score = (total - min_gain) / (max_gain - min_gain)

    return score


def score_metier_max(y_pred_proba, y_true, verbose=True):
    '''Créer un graphique permettant de déterminer quel peut être le score max en fonction du threshold.
    :param: y_pred_proba (probabilités prédites par le modèle), y_true (vraies valeurs),
    verbose (affiche le graphe ou juste le score maximal)
    :return: graphique,
    '''
    scores = []
    for threshold in np.linspace(0, 1, num=101):
        y_pred = np.where(y_pred_proba > threshold, 1, 0)
        score = fonction_metier(y_true, y_pred)
        scores.append(score)

    if verbose:
        score_max = max(scores)
        opti_threshold = np.linspace(0, 1, num=101)[scores.index(score_max)]
        y_pred = np.where(y_pred_proba > opti_threshold, 1, 0)

        print("Score métier maximum : {:.2f}".format(score_max))
        print("Threshold optimal    : {}".format(opti_threshold))

        fig, ax = plt.subplots(figsize=(6, 5))
        plt.plot(np.linspace(0, 1, num=101), scores, label="model score")
        plt.axvline(x=opti_threshold, color='k', dashes=(0.5, 1),
                    label="optimal threshold: {}".format(opti_threshold))

        plt.title("Score métier en fonction du threshold", fontsize=10)
        plt.xlabel("Thresholds", fontsize=10)
        plt.ylabel("Score métier", fontsize=10)
        plt.legend()
        plt.show()

    else:
        return max(scores)


import pickle


def display_shape(data_list: list, space=10):
    for df in data_list:
        print('{:{}}: {}'.format(namestr(df), space, df.shape))


def save_pickle(data: pd.DataFrame, filename: str, dossier: str):
    if not os.path.exists(dossier):
        os.makedirs(dossier)
    chemin = os_path_join(dossier, filename)
    with open(chemin, 'wb') as file:
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)


def load_pickle(file_path: str):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def load_csv(filename: str, index_var='SK_ID_CURR') -> pd.DataFrame:
    if index_var:
        data = pd.read_csv(f'{OUT_FOLDER}/{filename}').set_index(index_var)
    else:
        data = pd.read_csv(f'{OUT_FOLDER}/{filename}')
    return data


def sauvegarde_data(data, file_name, path=OUT_FOLDER, ):
    savepath = f'{OUT_FOLDER}/{file_name}'
    print('saving data')
    with timer(f'Save to {savepath}'):
        data.to_csv(savepath,
                    # index=False,
                    )


def list_pickles_files(directory: str, prefix: str = None):
    pickle_files = [filename for filename in os.listdir(directory)
                    if filename.endswith('.pickle')]

    if prefix is not None:
        pickle_files = [filename for filename in pickle_files
                        if filename.startswith(prefix)]

    return pickle_files
