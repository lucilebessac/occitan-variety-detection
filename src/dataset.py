import re
import os
import csv

def charger_corpus(dossier_data):
    corpus = []
    labels = []

    for fichier in os.listdir(dossier_data):
        if fichier.endswith(".csv"):
            chemin = os.path.join(dossier_data, fichier)

            with open(chemin, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="§")
                for ligne in reader : 
                    if len(ligne) == 2 and ligne[1].strip(): # Pour ignorer la ligne si la case label est vide
                        corpus.append(ligne[0])  #Texte
                        labels.append(int(ligne[1]))  #Label (0 ou 1)
                    else:
                        print(f"Ligne ignorée (mal formée) dans {fichier} : {ligne}") 

    return corpus, labels

def tokenizer_occitan(texte):

    texte = re.sub('’', "'", texte) 
    texte = re.sub(r"([a-zàèòáéíóúïüç])\-([nzu])\-([a-zàèòáéíóúïüç])", r"\1 - \2 - \3", texte, flags=re.IGNORECASE)
    texte = re.sub(r"\b([dlmnstçcbzu]')([a-zàèòáéíóúïüç])", r"\1 \2", texte, flags=re.IGNORECASE)
    texte = re.sub(r"\b((an|aquest|ent)')([a-zàèòáéíóúïüç])", r"\1 \3", texte, flags=re.IGNORECASE)
    texte = re.sub(r"\b((qu|ns|vs)')([a-zàèòáéíóúïüç])", r"\1 \3", texte, flags=re.IGNORECASE)
    texte = re.sub(r"(-(vos|nos|ne|se|te|la|lo|li|las|los|me|u|ac|i|lor|o))", r" \1", texte, flags=re.IGNORECASE)
    texte = re.sub(r"([a-zàèòáéíóúïüç])('(u|us|n|v|ns|vs|m|t|i|s|ac))", r"\1 \2", texte, flags=re.IGNORECASE)
    texte = re.sub(r"([!\"#$%&()*+,./:;<=>?@[\]^_`{|}~«»]{2,})", r" \1 ", texte, flags=re.IGNORECASE)
    texte = re.sub(r"(\w)([!\"#$%&()*+,./:;<=>?@[\]^_`{|}~«»])", r"\1 \2", texte, flags=re.IGNORECASE)
    texte = re.sub(r"([!\"#$%&()*+,./:;<=>?@[\]^_`{|}~«»])(\w)", r" \1 \2", texte, flags=re.IGNORECASE)
    tokens = re.split(r"\s+", texte.strip())

    return tokens