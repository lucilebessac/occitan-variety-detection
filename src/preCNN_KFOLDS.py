import csv 
import os
import re
import fasttext.util
from sklearn.model_selection import KFold
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

    # Validation croisée  
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    validation_scores = []

    for fold, (train_index, val_index) in enumerate(kfold.split(corpus)):
        print(f"Fold {fold + 1}") 
        
        # Diviser les données en train/validation pour ce fold
        X_train, X_val = np.array(corpus)[train_index], np.array(corpus)[val_index]
        y_train, y_val = np.array(labels)[train_index], np.array(labels)[val_index]

        # Limiter la longueur des phrases (99e percentile) (car problème de RAM)
        longueurs = [len(tokenizer_occitan(texte)) for texte in X_train] + [len(tokenizer_occitan(texte)) for texte in X_val]
        max_len = int(np.percentile(longueurs, 99))
        print(f"Longueur maximale après percentile 99% : {max_len}")

        # Tokenisation et Vectorisation avec FastText Occitan
        phrases_vectorisees_train = []
        for texte in X_train:
            phrase_tokenisee = tokenizer_occitan(texte)
            phrase_vectorisee = [model.get_word_vector(mot) for mot in phrase_tokenisee][:max_len]
            phrases_vectorisees_train.append(phrase_vectorisee)
        print(f"X_train est tokenisée et vectorisée.")  # Test
        
        phrases_vectorisees_val = []
        for texte in X_val:
            phrase_tokenisee = tokenizer_occitan(texte)
            phrase_vectorisee = [model.get_word_vector(mot) for mot in phrase_tokenisee][:max_len]
            phrases_vectorisees_val.append(phrase_vectorisee)
        print(f"X_val est tokenisée et vectorisée.")

        # Padding
        phrases_vectorisees_train_padded = pad_sequences(phrases_vectorisees_train, maxlen=max_len, dtype="float32", padding="post")
        phrases_vectorisees_val_padded = pad_sequences(phrases_vectorisees_val, maxlen=max_len, dtype="float32", padding="post")
        print(f"Le padding a été effectué.")

        # Conversion en tensors exploitables par Pytorch
        X_train_tensor = torch.tensor(phrases_vectorisees_train_padded, dtype=torch.float32)
        X_val_tensor = torch.tensor(phrases_vectorisees_val_padded, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        print(f"Conversion en tensor effectuée.")  # Test

        ## (à compléter)
        # Implémenter le CNN
  
        # Entrainer le CNN 
        #model_cnn.fit(blablablabla)
        
        break #faire qu'un split pour le test de lancement



if __name__ == "__main__":
    main()

