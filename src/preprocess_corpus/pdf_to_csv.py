import fitz  # module pour ouvrir et lire les fichiers PDF
import csv
import re
import os

def decouper_en_phrases(texte):
    # Séparer sur un caractère de ponctuation de fin de phrase ou un saut de ligne ou une fin de ligne
    return re.split(r'(?<=[.!?:])\s+', texte)

def pdf_to_csv(pdf_path, output_path):
    pdf_doc = fitz.open(pdf_path)
    
    if not output_path.endswith(".csv"):
        output_path += ".csv"
        
    with open(output_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter="§")
        
        all_phrases = []
        for page in pdf_doc:
            text = page.get_text("text")
            phrases = decouper_en_phrases(text)
            for phrase in phrases:
                phrase = phrase.strip().replace("\n", " ")
                if phrase:  # Check if phrase is not empty
                    all_phrases.append(phrase)
        
        # Pour skip les 2 premières lignes
        for phrase in all_phrases: #[2:]
            writer.writerow([phrase])  # LABEL À CHANGER
                    
    print("Fichier csv crée :", output_path)

def process_directory(pdf_dir, output_dir):
    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, file)
            output_path = os.path.join(output_dir, file.replace(".pdf", ".csv"))
            pdf_to_csv(pdf_path, output_path)

# Appel fonction
# pour tous les fichiers pdf dans le chemin donné
pdf_dir = "../data/"  # PATH À CHANGER
output_dir = "../data/"  # PATH À CHANGER
process_directory(pdf_dir, output_dir)
