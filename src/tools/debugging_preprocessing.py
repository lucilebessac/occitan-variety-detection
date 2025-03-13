from dataset import tokenizer_occitan


import fasttext.util
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

def main() : 


    texte = "L'ostal es blanc."
    # Télécharger le modèle FastText occitan
    fasttext.util.download_model('oc', if_exists='ignore')
    model = fasttext.load_model('cc.oc.300.bin') 

    max_len = 53

    # Tokeniser et Vectoriser
    phrases_vectorisees_train = []
    phrase_tokenisee = tokenizer_occitan(texte)
    print("phrase_tokenisee:", phrase_tokenisee)
    print("phrase_tokenisee shape:", len(phrase_tokenisee))
    phrase_vectorisee = [model.get_word_vector(mot) for mot in phrase_tokenisee][:max_len]
    print("phrase_vectorisee shape:", np.array(phrase_vectorisee).shape)
    phrases_vectorisees_train.append(phrase_vectorisee)
    print("phrases_vectorisees_train shape:", np.array(phrases_vectorisees_train).shape)
    print(f"X_train est tokenisée et vectorisée.") # Test
    

    # Padding
    phrases_vectorisees_train_padded = pad_sequences(phrases_vectorisees_train, maxlen=max_len, dtype="float16", padding="post") 
    print("phrases_vectorisees_train_padded shape:", phrases_vectorisees_train_padded.shape)
    print(f"Le padding a été effectué.") # Test

    # Convertir en tensors exploitables par Pytorch
    X_train_tensor = torch.tensor(phrases_vectorisees_train_padded, dtype=torch.float32)
    print("X_train_tensor shape:", X_train_tensor.shape)
    print(f"Conversion en tensor effectuée.") # Test

    # Faire des batches 
    train_data = TensorDataset(X_train_tensor) 
    #shape train data
    print("train_data shape:", train_data.tensors[0].shape)

    batch_size = 16 
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size) 
    print("train_loader shape:", train_loader.dataset.tensors[0].shape)

if __name__ == "__main__":
    main()


#     phrase_tokenisee: ["L'", 'ostal', 'es', 'blanc', '.']
# phrase_tokenisee shape: 5
# phrase_vectorisee shape: (5, 300)
# phrases_vectorisees_train shape: (1, 5, 300)
# X_train est tokenisée et vectorisée.
# phrases_vectorisees_train_padded shape: (1, 53, 300)
# Le padding a été effectué.
# X_train_tensor shape: torch.Size([1, 53, 300])
# Conversion en tensor effectuée.
# train_data shape: torch.Size([1, 53, 300])
# train_loader shape: torch.Size([1, 53, 300])
