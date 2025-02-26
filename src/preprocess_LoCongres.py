import csv
import os

def recup_2_premieres_colonnes(LoCongres_dossier, output_dossier):
    """
    Cette fonction garde les deux premières colonnes des csv récupérés sur LoCongres.
    """

    for file in os.listdir(LoCongres_dossier) : 
        if file.endswith(".csv"):

            input_path = os.path.join(LoCongres_dossier, file)
            output_path = os.path.join(output_dossier, file)

            with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8", newline="") as outfile:
                reader = csv.reader(infile, delimiter="§")
                writer = csv.writer(outfile, delimiter="§")

                for ligne in reader:
                    writer.writerow(ligne[:2])

            print("Fichier csv crée :", output_path)


#Appel fonction
LoCongres_dossier = "/Users/manongourves/Desktop/Master_TAL/M2/S2/neural_net/projet/data/gascon"
output_dossier = "/Users/manongourves/Desktop/Master_TAL/M2/S2/neural_net/projet/data_OK/gascon"
recup_2_premieres_colonnes(LoCongres_dossier, output_dossier)