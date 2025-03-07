# script qui lit tous les fichiers .csv d'un dossier
# si la 2e case de chaque colonne ne contient pas "oc-lengadoc-grclass", on ajoute "oc-lengadoc-grclass" dans la 2e case
# le séparateur est §

import csv
import os

def ajout_tag(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file, delimiter="§")
        with open(output_path, "w", encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file, delimiter="§")
            for row in reader:
                # si il n'y a pas de deuxième case ou si la deuxième case n'est pas "oc-lengadoc-grclass"
                if len(row) < 2 or row[1] != "oc-lengadoc-grclass":
                    row.insert(1, "oc-lengadoc-grclass")
                writer.writerow(row)
    print("Fichier csv crée :", output_path)

def process_directory(input_dir, output_dir):
    for file in os.listdir(input_dir):
        if file.endswith(".csv"):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, file.replace(".csv", "_oc-lengadoc-grclass.csv"))
            ajout_tag(input_path, output_path)

input_dir = "../../data/V3/lengadocian/"  # PATH À CHANGER
output_dir = "../../data/V3/lengadocian/clean/"  # PATH À CHANGER
process_directory(input_dir, output_dir)
