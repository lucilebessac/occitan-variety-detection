import csv 
import os
import re
import fasttext.util
from sklearn.model_selection import train_test_split
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import torch

def charger_corpus(dossier_data):
    corpus = []
    labels = []

    for fichier in os.listdir(dossier_data):
        if fichier.endswith(".csv"):
            chemin = os.path.join(dossier_data, fichier)

            with open(chemin, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="§")
                for ligne in reader : 
                    if len(ligne) == 2:
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

def main() : 

    dossier_data = "../../DATA_OK" ## PATH À ADAPTER

    # Télécharger le modèle FastText occitan
    fasttext.util.download_model('oc', if_exists='ignore')
    model = fasttext.load_model('cc.oc.300.bin') 

    # Charger le corpus, obtenir le texte et les labels
    corpus, labels = charger_corpus(dossier_data)

    # Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(corpus, labels, test_size=0.2, random_state=42)
    print(f"Nombre d'exemples dans l'ensemble d'entraînement : {len(X_train)}") #Test
    print(f"Nombre d'exemples dans l'ensemble de test : {len(X_test)}") #Test

    # Limiter la longueur des phrases (99e percentile) (car problème de RAM)
    longueurs = [len(tokenizer_occitan(texte)) for texte in X_train + X_test]
    max_len = int(np.percentile(longueurs, 99))
    print(f"Longueur maximale après percentile 99% : {max_len}") #Test

    # Tokenisation et Vectorisation
    phrases_vectorisees_train = []
    for texte in X_train:
        phrase_tokenisee = tokenizer_occitan(texte)
        phrase_vectorisee = [model.get_word_vector(mot) for mot in phrase_tokenisee][:max_len]
        phrases_vectorisees_train.append(phrase_vectorisee)
    print(f"X_train est tokenisée et vectorisée.") # Test
    
    phrases_vectorisees_test = []
    for texte in X_test:
        phrase_tokenisee = tokenizer_occitan(texte)
        phrase_vectorisee = [model.get_word_vector(mot) for mot in phrase_tokenisee][:max_len]
        phrases_vectorisees_test.append(phrase_vectorisee)
    print(f"X_test est tokenisée et vectorisée.") # Test

    # Padding
    phrases_vectorisees_train_padded = pad_sequences(phrases_vectorisees_train, maxlen=max_len, dtype="float32", padding="post") 
    phrases_vectorisees_test_padded = pad_sequences(phrases_vectorisees_test, maxlen=max_len, dtype="float32", padding="post")
    print(f"Le padding a été effectué.") # Test

    # Conversion en tensors exploitables par Pytorch
    X_train_tensor = torch.tensor(phrases_vectorisees_train_padded, dtype=torch.float32)
    X_test_tensor = torch.tensor(phrases_vectorisees_test_padded, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    print(f"Conversion en tensor effectuée.") # Test

    #Couche d'embedding 

    #Couches convolutives

    #Couche max-pooling

    #Activation ReLU ?

    #Couche Softmax

    # Faire des batches (petits groupes d'exemples) pour l'entrainement 
        #utiliser DataLoader




if __name__ == "__main__":
    main()
