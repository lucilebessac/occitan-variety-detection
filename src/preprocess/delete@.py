## script qui lit un fichier csv, supprime toutes les lignes contenant un "@" et sauvegarde dans le même fichier

import os
import csv

def delete_at(input_path):
    with open(input_path, "r", encoding="utf-8") as infile:
        lignes = infile.readlines()
        lignes = [ligne for ligne in lignes if "@" not in ligne]
    
    with open (input_path, "w", encoding="utf-8") as outfile:
        for ligne in lignes:
            outfile.write(ligne)

input_path = "../../data/V3/gascon/oc-gascon-grclass.csv"
delete_at(input_path)
