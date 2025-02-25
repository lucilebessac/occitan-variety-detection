import fitz #module pour ouvrir et lire les fichiers PDF
import csv
import re

def decouper_en_phrases(texte):
    #Séparer sur ".", "?" ou "!" suivi d'un espace ou d'une fin de ligne
    return re.split(r'(?<=[.!?])\s+', texte)

def pdf_to_csv(pdf_path, output_path): 
    pdf_doc = fitz.open(pdf_path)

    if not output_path.endswith(".csv"):
        output_path += ".csv"

    with open(output_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter="§")
        writer.writerow(["Texte", "Label"])

        for page in pdf_doc:
            text = page.get_text("text")

            phrases = decouper_en_phrases(text)

            for phrase in phrases:
                phrase = phrase.strip().replace("\n", " ")
                writer.writerow([phrase, "oc-lengadoc-grclass"])   ##LABEL À CHANGER 
 

    print("Fichier csv crée :", output_path)


#Appel fonction
pdf_path = "/Users/manongourves/Desktop/Master_TAL/M2/S2/neural_net/projet/declaracion_universala_languedocien.pdf"
output_path = "/Users/manongourves/Desktop/Master_TAL/M2/S2/neural_net/projet/data_OK/lengadocian"
pdf_to_csv(pdf_path, output_path)





















