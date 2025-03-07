from dataset import charger_corpus, tokenizer_occitan
from occitanCNN import OccitanCNN
from utils import compter_labels

import fasttext.util
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

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

    # Limiter la longueur des phrases (car problème de RAM)
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

    # Faire des batches 
    train_data = TensorDataset(X_train_tensor, y_train_tensor) # Chaque item est un tuple
    test_data = TensorDataset(X_test_tensor, y_test_tensor)

    batch_size = 16 # Taille des petits groupes (batches) pour pas tout process d'un coup
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size) # Askip on Shuffle pour réduire les biais, ça parait logique mais j'ai pas compris en profondeur
    test_loader = DataLoader(test_data, batch_size=batch_size) # Pas besoin de shuffle pour le test
    print(f"DataLoaders créés.") # Test

    # Gestion du déséquilibre des classes
    class_weights = compter_labels(y_train)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Initialisation du modèle
    #model = OccitanCNN()

    # Lancer le train



    # Lancer l'évaluation



if __name__ == "__main__":
    main()
