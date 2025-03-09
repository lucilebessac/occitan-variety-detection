from dataset import charger_corpus, tokenizer_occitan
from occitanCNN import OccitanCNN
from utils import compter_labels, save_results
from training import train_model, evaluate_model

import fasttext.util
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

def main() : 

    dossier_data = "../data" ## PATH À ADAPTER

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
    max_len = int(np.percentile(longueurs, 99))
    print(f"Longueur maximale après percentile 99% : {max_len}") #Test

    # Tokeniser et Vectoriser
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

    # Convertir en tensors exploitables par Pytorch
    X_train_tensor = torch.tensor(phrases_vectorisees_train_padded, dtype=torch.float32)
    X_test_tensor = torch.tensor(phrases_vectorisees_test_padded, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    print(f"Conversion en tensor effectuée.") # Test

    # Faire des batches 
    train_data = TensorDataset(X_train_tensor, y_train_tensor) 
    test_data = TensorDataset(X_test_tensor, y_test_tensor)

    batch_size = 16 
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size) 
    test_loader = DataLoader(test_data, batch_size=batch_size) 
    print(f"DataLoaders créés.") # Test

    # Gérer le déséquilibre des classes
    class_weights = compter_labels(y_train, nbr_classes=3)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Initialisation du modèle
    model = OccitanCNN()
    print(f"Model Occitan CNN initialisé, lancement de l'entrainement.")

    # Lancer le train
    epochs = 20
    learning_rate = 0.0005
    train_model(model, train_loader, criterion, epochs=epochs, learning_rate=learning_rate)

    # Lancer l'évaluation
    print(f"Evaluation de Model Occitan CNN en cours.")
    avg_loss, accuracy, precision, recall, f1, classif_report = evaluate_model(model, test_loader, criterion)

    # Affichage des résultats
    print(f"\nRésultats de l'évaluation avec epoch={epochs} et learning rate={learning_rate} :")
    print(f"Average Loss: {avg_loss:.4f}")
    print("\nClassification Report:\n", classif_report)
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Enregistrement des résultats
    save_results(epochs, learning_rate, avg_loss, accuracy, precision, recall, f1, classif_report)
    print(f"Les résultats ont été sauvegardés dans le dossier results.")


if __name__ == "__main__":
    main()
