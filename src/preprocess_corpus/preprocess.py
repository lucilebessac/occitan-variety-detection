import csv 
import os 
import re 

def preprocess(dossier_data, output_data): 
    """
    Cette fonction applique les transformations suivantes sur un dossier de fichiers .csv :
        1. Change les labels : oc-gascon-grclass -> 0, oc-lengadoc-grclass -> 1
        2. Met le texte en minuscule.
        3. Enlève la ponctuation sauf les apostrophes.
        4. Enregistre dans un nouveau dossier.
    """
    if not os.path.exists(output_data) : 
        os.mkdir(output_data)

    for file in os.listdir(dossier_data) : 
        if file.endswith(".csv"):

            input_path = os.path.join(dossier_data, file)
            output_path = os.path.join(output_data, file)

            with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8", newline="") as outfile :
                reader = csv.reader(infile, delimiter="§")
                writer = csv.writer(outfile, delimiter="§")

                for ligne in reader : 

                    #vérifie si la ligne est bien formée (2 colonnes)
                    if len(ligne) != 2:
                        print(f"Ligne ignorée (mal formée) : {ligne}, fichier {infile}")
                        continue

                    #supprimer les espaces dans les colonnes
                    for i in range(len(ligne)):
                        ligne[i] = ligne[i].strip()

                    #changer le label
                    if ligne[1] == "oc-gascon-grclass" :
                        ligne[1] = "0"
                    elif ligne[1] == "oc-lengadoc-grclass" : 
                        ligne[1] = "1"
                    else:
                        print(f"Ligne ignorée (label inconnu) : {ligne}")
                        continue

                    #mettre en minuscule
                    ligne[0] = ligne[0].lower()

                    #enlever la ponctuation
                    ligne[0] = re.sub(r"[^\w\s']", "", ligne[0])

                    #écrit la ligne traitée
                    writer.writerow(ligne)
            
            print("Fichier traité :", output_path)
                    

#Appel fonction
dossier_data = "/Users/manongourves/Desktop/Master_TAL/M2/S2/neural_net/projet/all_data" ##PATH À ADAPTER
output_data = "/Users/manongourves/Desktop/Master_TAL/M2/S2/neural_net/projet/DATA_OK" ##PATH À ADAPTER
preprocess(dossier_data, output_data)