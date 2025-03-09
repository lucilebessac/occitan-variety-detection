## Code qui prend en entrée un fichier qui se termine par brut.csv
# lit chaque ligne de la colonne 1, avant le delimiter §
# si la ligne ne se termine pas par un point, on regarde si la suivante commence par une majuscule
# si c'est le cas, on ne concatène pas les deux lignes
# sinon, on concatène les deux lignes
# on écrit le résultat dans un fichier brut2.csv

import csv
import os

def concatener_phrases(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file, delimiter="§")
        with open(output_path, "w", encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file, delimiter="§")
            for row in reader:
                if row[0][-1] != ".":
                    # si la ligne suivante commence par une majuscule
                    if next is not None and next[0].islower():
                        writer.writerow([row[0] + " " + next])
    print("Fichier csv crée :", output_path)

def process_directory(input_dir, output_dir):
    for file in os.listdir(input_dir):
        if file.endswith("brut.csv"):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, file.replace("brut.csv", "brut2.csv"))
            concatener_phrases(input_path, output_path)

# Appel fonction
# pour tous les fichiers csv dans le chemin donné
input_dir = "../data/"  # PATH À CHANGER
output_dir = "../data/"  # PATH À CHANGER
process_directory(input_dir, output_dir)