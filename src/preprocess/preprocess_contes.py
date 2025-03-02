import os
import csv
import re

def decouper_en_phrases(texte):
    #Séparer sur ".", "?" ou "!" suivi d'un espace ou d'une fin de ligne
    return re.split(r'(?<=[.!?])\s+', texte)

def preprocess_contes(contes_dossier, output_dossier):
    """
    Cette fonction récupère la deuxième colonne des contes .txt, segmente par phrase en ajoutant un label et enregistre un csv. 
    """
    
    for file in os.listdir(contes_dossier) : 
        if file.endswith(".txt"):

            input_path = os.path.join(contes_dossier, file)
            output_file = file.replace(".txt", ".csv")
            output_path = os.path.join(output_dossier, output_file)

            #créer un csv avec § comme séparateur
            with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
                writer = csv.writer(outfile, delimiter="§")

                mots = []

                #choper la 2eme colonne 
                for ligne in infile : 
                    colonnes = ligne.strip().split("\t")
                    if len(colonnes) > 1:
                        mots.append(colonnes[1])
                
                texte_complet = " ".join(mots)
    
                #split sur [.!?] et j'ajoute §oc-lengadoc-grclass après la ponctuation
                phrases = decouper_en_phrases(texte_complet)

                for phrase in phrases : 
                    phrase = phrase.strip()
                    writer.writerow([phrase, "oc-lengadoc-grclass"])
                    
            #enregistrer dans le dossier "output_dossier"
            print("Fichier csv crée :", output_path)


##Appel fonction
contes_dossier = "/Users/manongourves/Desktop/Master_TAL/M2/S2/neural_net/projet/data/gascon" ##PATH À CHANGER
output_dossier = "/Users/manongourves/Desktop/Master_TAL/M2/S2/neural_net/projet/data_OK/gascon" ##PATH À CHANGER
preprocess_contes(contes_dossier, output_dossier)