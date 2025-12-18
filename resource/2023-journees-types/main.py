import calendar
import locale
import datetime
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


# entree : le nombre de colonnes dans le dataset
# sortie : une liste de 7 listes qui contiennent elles-memes nCol listes vides
def creation_liste(nCol):
    liste = []
    for i in range(7):
        liste.append([])
        for j in range(nCol):
            liste[i].append([])
    return liste


# entree : les donnees issues du fichier csv
# sortie : une liste de 7 dataframes : 1 par jour de la semaine
def decoupage(data):
    premierTimestamp = data.loc[0, "Timestamp"]
    premierDate = datetime.datetime.fromtimestamp(premierTimestamp)  # conversion du premier Timestamp en date
    premierJour = premierDate.weekday()  # monday 0 --> sunday 6
    heureDebut = premierDate.time().strftime('%H:%M:%S')  # la premiere heure enregistree dans les donnees
    t0 = premierTimestamp  # t0 = timestamp à minuit le premier jour des donnees
    if heureDebut != "00:00:00":  # donc si t0 ne correspond pas a minuit
        premiereTimezone = data.loc[0, "Timezone : Europe/Paris"]
        t0 -= (int(premiereTimezone[11:13])*60*60 + int(premiereTimezone[14:16])*60 + int(premiereTimezone[17:19]))  # decalage a minuit

    colonnes = data.columns
    nbColonnes = len(colonnes)
    listesDonnees = creation_liste(nbColonnes)  # cree la liste qui recueille ensuite les donnees par jour et par type de donnees

    for i in range(len(data)):  # parcours de chaque ligne du dataset
        numJour = ((((data.iloc[i, 0] - t0) % 604800) // 86400) + premierJour) % 7  # calcul du jour
        for j in range(nbColonnes):  # parcours de chaque donnée (colonne) dans cette ligne de données
            listesDonnees[numJour][j].append(data.iloc[i, j])  # ajout de chaque donnée dans la liste correspondante de listesDonnees

    dataframes = []
    for jour in range(7):  # pour chaque jour de la semaine
        dico = {}
        for k in range(nbColonnes):  # remplit le dictionnaire contenant toutes les donnees de ce jour de la semaine
            dico[colonnes[k]] = listesDonnees[jour][k]
        dataframe = pd.DataFrame(dico)  # puis construit le dataframe correspondant
        dataframes.append(dataframe)  # avant de l'ajouter a une liste de dataframes que l'on retourne

    return dataframes


# entree : le temps en minutes a convertir en angle
# sortie : l'angle correspondant au temps en entree
def conversion_minutes_angle(temps):
    return temps * (2 * math.pi) / 1440


# entree : l'angle en radians a convertir en heures
# sortie : l'heure correspondante a l'angle en entree
def conversion_angle_heures(angle):
    return angle * 24 / (2 * math.pi)


# entree : la liste des temps, la liste des donnees, la valeur choisie pour kappa, le reechantillonage choisi en minutes
# sortie : une liste contenant les temps cible, une liste contenant les donnees lissees
def attribution_poids(lstTemps, lstDonnees, k, echantillon):
    resTemps = []
    resDonnees = []
    tpsMin = min(lstTemps)
    for i in range(len(lstTemps)):
        lstTemps[i] = lstTemps[i] - tpsMin  # initialisation du temps à 0
    tc = 0  # initialisation du temps cible qui varie par pas de echantillon
    pasAngulaire = conversion_minutes_angle(echantillon)  # conversion du pas
    while tc <= 2 * math.pi:  # parcours du cercle entier (donc d'une journee entiere)
        sommePond = 0
        sommeCoef = 0
        for j in range(len(lstTemps)):  # parcours de toutes les donnees
            if hasattr(lstDonnees[j], '__iter__'):
                lstDonnees[j] = float(lstDonnees[j].replace(",", "."))
            if not np.isnan(lstDonnees[j]):  # si la donnee est un nombre
                t = conversion_minutes_angle(lstTemps[j]/60)   # conversion de son timestamp en angle
                poids = math.exp(k * math.cos(t - tc))  # application de la loi de Von Mises pour connaitre le poids de la donnee dans la moyenne ponderee
                sommePond += lstDonnees[j] * poids  # ajout de la donnee multipliee par son poids a la somme ponderee
                sommeCoef += poids  # ajout du poids a la somme des poids
        if sommeCoef == 0:  # si la somme des poids vaut 0, c'est qu'il n'y avait pas de nombre dans la fenetre autour du temps cible
            valeurLisseeTC = np.nan  # donc la valeur du temps cible lissee n'est pas un nombre
        else:
            valeurLisseeTC = sommePond / sommeCoef  # sinon la valeur du temps cible lissee est la moyenne ponderee
        resTemps.append(tc)  # ajout du temps cible a la liste des nouveaux temps
        resDonnees.append(valeurLisseeTC)  # ajout de la donnee lissee correspondante dans la liste des nouvelles donnees
        tc += pasAngulaire  # incrementation : on passe au temps cible suivant
    return [resTemps, resDonnees]


def graphique(projection, x, y, numeroJour=None, typeDonnees=None):
    echelleAxe = (max(y) - min(y)) / 10
    minAxe = min(y) - echelleAxe
    maxAxe = max(y) + echelleAxe
    if projection == "polaire":
        plt.axes(projection='polar')
        plt.title(f"{calendar.day_name[numeroJour]}-type : {typeDonnees}")
        plt.ylim(minAxe, maxAxe)
        plt.polar(x, y, lw=1)  # , color=''
        plt.show()
    else:
        for i in range(len(x)):
            x[i] = conversion_angle_heures(x[i])
        fig, ax = plt.subplots()
        plt.title(f"{calendar.day_name[numeroJour]}-type : {typeDonnees}")
        ax.plot(x, y, lw=1)
        ax.set(xlim=(0, 24), xticks=np.arange(0, 25, 2), ylim=(minAxe, maxAxe))
        plt.show()


if __name__ == '__main__':

    locale.setlocale(locale.LC_ALL, 'french')

    # etape 0 : initialisation des parametres et lecture du fichier
    fichier = "data/Salon_01_02_2017.csv"  # Salon_01_02_2017.csv  RL2018_Fevrier_Timestamp.csv  bureau_16_1_2019.csv
    separateur = ";"                       # Douche_01_02_2017.csv  Outside_01_02_2017.csv
    ligneHeader = 3

    kappa = 10
    numeroJour = 1
    reechantillonage = 10
    typeDonnees = "CO2"

    typeProjection = "polaire"

    donnees = pd.read_csv(fichier, sep=separateur, header=ligneHeader, engine="python")

    # etape 1 : decoupage des points initiaux en troncons
    troncons = decoupage(donnees)  # troncons contient 7 dataframes : de lundi a dimanche
    troncon = troncons[numeroJour]
    print(f"Dataset {calendar.day_name[numeroJour]} :\n", troncon)

    # etape 2 : application de la fonction de repartition des poids et calcul des moyennes ponderees glissantes
    listeTemps = troncon.loc[:, "Timestamp"].to_list()
    listeDonnees = troncon.loc[:, typeDonnees].to_list()
    resultatsLisses = attribution_poids(listeTemps, listeDonnees, kappa, reechantillonage)
    # resultatsLisses = moyenneSimple(listeTemps, listeDonnees)
    tempsLisses = resultatsLisses[0]
    donneesLissees = resultatsLisses[1]
    print(f"\nTemps : Données lissées pour {typeDonnees}")
    for ind in range(len(tempsLisses)):
        print(f"{tempsLisses[ind]} : {donneesLissees[ind]}")

    # etape 3 : affichage sous la forme d'un graphique
    graphique(typeProjection, tempsLisses, donneesLissees)
