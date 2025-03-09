import os
import csv
import re

def decouper_en_phrases(texte):
    return texte.split('\n')

def preprocess_simple_txt(txt_dossier, output_dossier):
    """
    Lit des fichiers txt somples (sans tab ou autre comme les contes), segmente par lignes, ajoute un label et enregistre un csv.
    """
    os.makedirs(output_dossier, exist_ok=True)
    
    for file in os.listdir(txt_dossier):
        if file.endswith(".txt"):
            input_path = os.path.join(txt_dossier, file)
            output_file = file.replace(".txt", ".csv")
            output_path = os.path.join(output_dossier, output_file)
            
            # créer un csv avec § comme séparateur
            with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8", newline='') as outfile:
                writer = csv.writer(outfile, delimiter="§")                
                # lire le fichier et diviser par lignes
                for ligne in infile:
                    ligne = ligne.strip()
                    if ligne:  #  ignore lignes vides
                        writer.writerow([ligne, "oc-lengadoc-grclass"])
                
            # enregistrer dans le dossier "output_dossier"
            print("Fichier csv crée :", output_path)

# Appel fonction
txt_dossier = "../data/"  # PATH À CHANGER
output_dossier = "../data/clean/"  # PATH À CHANGER
preprocess_simple_txt(txt_dossier, output_dossier)