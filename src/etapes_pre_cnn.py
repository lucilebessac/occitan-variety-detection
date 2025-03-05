import csv 
import os
import re
import fasttext.util
from sklearn.model_selection import KFold
import numpy as np
from keras.preprocessing.sequence import pad_sequences

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
        
        X_train, X_val = np.array(corpus)[train_index], np.array(corpus)[val_index]
        y_train, y_val = np.array(labels)[train_index], np.array(labels)[val_index]

        # Tokeniser le corpus
        phrases_tokenisees_train = [tokenizer_occitan(texte) for texte in X_train]
        phrases_tokenisees_val = [tokenizer_occitan(texte) for texte in X_val]

        print(f"Exemple phrase tokenisé (train) : {phrases_tokenisees_train[0]}") #Test

        # Vectorisation avec FastText Occitan
        phrases_vectorisees_train = [[model.get_word_vector(mot) for mot in phrase] for phrase in phrases_tokenisees_train]
        phrases_vectorisees_val = [[model.get_word_vector(mot) for mot in phrase] for phrase in phrases_tokenisees_val]

        print(f"Taille phrase vectorisée (train) : {len(phrases_vectorisees_train[0])}") #Test
        print(f"Exemple vecteur : {phrases_vectorisees_train[0][0][:5]}...") #Test

        # Padding
        longueur_max = 0
        for phrase in phrases_vectorisees_train + phrases_vectorisees_val:
            if len(phrase) > longueur_max:
                longueur_max = len(phrase)

        phrases_vectorisees_train_padded = pad_sequences(phrases_vectorisees_train, maxlen=longueur_max, dtype='float16', padding='post') #float16 car pas assez de RAM
        phrases_vectorisees_test_padded = pad_sequences(phrases_vectorisees_val, maxlen=longueur_max, dtype='float16', padding='post')

        print(f"Longueur maximale après padding : {longueur_max}") #Test
        print(f"Exemple après padding : {phrases_vectorisees_train_padded[0][:5]}...") #Test

        ## (à compléter)
        # Implémenter le CNN
  
        # Entrainer le CNN 
        #model_cnn.fit(blablablabla)
        
        break #faire qu'un split pour le test 



if __name__ == "__main__":
    main()

