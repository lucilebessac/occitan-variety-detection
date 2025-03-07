import csv 
import os
import re
import fasttext.util
import torch

import torch.nn as nn
import numpy as np

from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader

# Dans les tutos ils font des classes pour les modèles alors j'ai fait une classe
# Pour les embeddings j'ai mis 300 car c'est la taille des embeddings de FastText
# nb_filtres = nb motifs de convolution à apprendre. 100 pk pas ?

class OccitanCNN(nn.Module):
    def __init__(self, fasttext_embedding_dim=300, nb_filtres=100):
        super(OccitanCNN, self).__init__()


    #Couche d'embedding 

        # Si je comprends bien, pas besoin de re faire l'embedding, on peut utiliser celui de FastText ?
        # Aaah on en a parlé mais je me souviens pas, au pire ce sera une ligne à ajouter

    #Couches convolutives

        # Couches convolutives avec différentes tailles de filtres (kernels) : Bi-grams, tri_grams etc
        self.conv1 = nn.Conv1d(in_channels=fasttext_embedding_dim, out_channels=nb_filtres, kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=fasttext_embedding_dim, out_channels=nb_filtres, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=fasttext_embedding_dim, out_channels=nb_filtres, kernel_size=4)

        # On peut faire du "dropout" pour éviter l'overfitting . On désactive des neurones aléatoirement avec une proba choisie : ici 0.5
        # self.dropout = nn.Dropout(0.5)

        # Couche "fully connected layer" pour la classification
        self.fc = nn.Linear(nb_filtres*3, 3) # *3 car on a 3 couches, puis 3 car on a 3 classes ? Enfin là dans les labels on en a que 2 mais on en veut 3 à la fin donc jsp

    def forward(self, x):
        # x est un tensor de taille (batch_size, max_len, fasttext_embedding_dim)
        # On veut un tensor de taille (batch_size, nb_filtres*3) à la fin
        x = x.permute(0, 2, 1) # On permute les dimensions pour que la taille des embeddings soit en 2e position. C'est pour que ce soit compatible avec PyTorch
        
    #Activation ReLU ?

        # On applique les convolutions + ReLu
        conv1 = nn.functional.relu(self.conv1(x))
        conv2 = nn.functional.relu(self.conv2(x))
        conv3 = nn.functional.relu(self.conv3(x))   

    # Max pooling

        # On applique le max pooling sur les convolutions
        pool1 = nn.functional.max_pool1d(conv1, conv1.size(2)).squeeze(2) # Là j'avoue il est tard et j'ai pas compris
        pool2 = nn.functional.max_pool1d(conv2, conv2.size(2)).squeeze(2)
        pool3 = nn.functional.max_pool1d(conv3, conv3.size(2)).squeeze(2)

    # Concaténation

        # On concatène les max pooling
        x = torch.cat((pool1, pool2, pool3), dim=1)

    # Dropout

        # On applique le dropout
        # x = self.dropout(x)

    # Couche fully connected
    
        # On applique la couche fully connected
        x = self.fc(x)

    #Couche Softmax

        # On applique la fonction softmax pour obtenir des probabilités
        x = nn.functional.softmax(x, dim=1)

        return x

def train_model(model, train_loader, test_loader, epochs=5, learning_rater=0.001): # CHOISIR LES BONNES VALEURS
    """

    ......

    """

def evaluate_model(model, test_loader):
    """
    
    ......

    """


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
    max_len = int(np.percentile(longueurs, 90))
    print(f"Longueur maximale après percentile 90% : {max_len}") #Test

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
    phrases_vectorisees_train_padded = pad_sequences(phrases_vectorisees_train, maxlen=max_len, dtype="float16", padding="post") 
    phrases_vectorisees_test_padded = pad_sequences(phrases_vectorisees_test, maxlen=max_len, dtype="float16", padding="post")
    print(f"Le padding a été effectué.") # Test

    # Conversion en tensors exploitables par Pytorch
    X_train_tensor = torch.tensor(phrases_vectorisees_train_padded, dtype=torch.float16)
    X_test_tensor = torch.tensor(phrases_vectorisees_test_padded, dtype=torch.float16)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    print(f"Conversion en tensor effectuée.") # Test


    # Faire des batches (petits groupes d'exemples) pour l'entrainement 
        #utiliser DataLoader
    train_data = TensorDataset(X_train_tensor, y_train_tensor) # Chaque item est un tuple
    test_data = TensorDataset(X_test_tensor, y_test_tensor)

    batch_size = 16 # Taille des petits groupes (batches) pour pas tout process d'un coup
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size) # Askip on Shuffle pour réduire les biais, ça parait logique mais j'ai pas compris en profondeur
    test_loader = DataLoader(test_data, batch_size=batch_size) # Pas besoin de shuffle pour le test
    print(f"DataLoaders créés.") # Test

# Lancer le train

# Lancer l'évaluation

if __name__ == "__main__":
    main()
