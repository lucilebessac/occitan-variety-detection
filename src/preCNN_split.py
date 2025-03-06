import csv 
import os
import re
import fasttext.util
from sklearn.model_selection import train_test_split
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

    # Tokeniser le corpus
    phrases_tokenisees_train = [tokenizer_occitan(texte) for texte in X_train]
    phrases_tokenisees_test = [tokenizer_occitan(texte) for texte in X_test]

    print(f"Exemple phrase tokenisé (train) : {phrases_tokenisees_train[0]}") # Test

    # Vectorisation avec FastText Occitan
    phrases_vectorisees_train = [[model.get_word_vector(mot) for mot in phrase] for phrase in phrases_tokenisees_train]
    phrases_vectorisees_test = [[model.get_word_vector(mot) for mot in phrase] for phrase in phrases_tokenisees_test]

    print(f"Taille phrase vectorisée (train) : {len(phrases_vectorisees_train[0])}") # Test
    print(f"Exemple vecteur : {phrases_vectorisees_train[0][0][:5]}...") # Test

    # Padding
    longueur_max = 0
    for phrase in phrases_vectorisees_train + phrases_vectorisees_test:
        if len(phrase) > longueur_max:
            longueur_max = len(phrase)

    phrases_vectorisees_train_padded = pad_sequences(phrases_vectorisees_train, maxlen=longueur_max, dtype="float16", padding="post") 
    phrases_vectorisees_test_padded = pad_sequences(phrases_vectorisees_test, maxlen=longueur_max, dtype="float16", padding="post")

    print(f"Longueur maximale après padding : {longueur_max}") # Test
    print(f"Exemple après padding : {phrases_vectorisees_train_padded[0][:5]}...") # Test

    # Conversion en tensors exploitables par Pytorch
    X_train_tensor = torch.tensor(phrases_vectorisees_train_padded, dtype=torch.float16)
    X_test_tensor = torch.tensor(phrases_vectorisees_test_padded, dtype=torch.float16)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    print(f"Shape tensor {X_train_tensor.shape}") # Test

    # Faire des batches (petits groupes d'exemples)



    ## (à compléter) CNN etc.


if __name__ == "__main__":
    main()
