from dataset import charger_corpus, tokenizer_occitan, vectorizer_phrase, padding_liste_phrases, tensorizer_phrase, charger_fasttext   
from occitanCNN import OccitanCNN
from utils import compter_labels, save_results
from training import train_model, evaluate_model

import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

def main() : 

    dossier_data = "../data" ## PATH À ADAPTER

    # Télécharger le modèle FastText occitan
    model = charger_fasttext()

    # Charger le corpus, obtenir le texte et les labels
    corpus, labels = charger_corpus(dossier_data)

    # Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(corpus, labels, test_size=0.2, random_state=42)
    print(f"Nombre d'exemples dans l'ensemble d'entraînement : {len(X_train)}") #Test
    print(f"Nombre d'exemples dans l'ensemble de test : {len(X_test)}") #Test

    # Limiter la longueur des phrases (car problème de RAM)
    longueurs = [len(tokenizer_occitan(texte)) for texte in X_train + X_test]
    max_len = int(np.percentile(longueurs, 99)) #max_len = 53
    print(f"Longueur maximale après percentile 99% : {max_len}") #max_len = 53

    # Tokeniser et Vectoriser
    phrases_vectorisees_train = []
    for texte in X_train:
        phrase_tokenisee = tokenizer_occitan(texte)
        phrase_vectorisee = vectorizer_phrase(phrase_tokenisee, model, max_len)
        phrases_vectorisees_train.append(phrase_vectorisee)
    print(f"X_train est tokenisée et vectorisée.") # Test
    
    phrases_vectorisees_test = []
    for texte in X_test:
        phrase_tokenisee = tokenizer_occitan(texte)
        phrase_vectorisee = vectorizer_phrase(phrase_tokenisee, model, max_len)
        phrases_vectorisees_test.append(phrase_vectorisee)
    print(f"X_test est tokenisée et vectorisée.") # Test

    # Padding
    phrases_vectorisees_train_padded = padding_liste_phrases(phrases_vectorisees_train, max_len)
    phrases_vectorisees_test_padded = padding_liste_phrases(phrases_vectorisees_test, max_len)
    print(f"Le padding a été effectué.") # Test

    # Convertir en tensors exploitables par Pytorch
    X_train_tensor = tensorizer_phrase(phrases_vectorisees_train_padded, dtype=torch.float32)
    X_test_tensor = tensorizer_phrase(phrases_vectorisees_test_padded, dtype=torch.float32)
    y_train_tensor = tensorizer_phrase(y_train, dtype=torch.long)
    y_test_tensor = tensorizer_phrase(y_test, dtype=torch.long)
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
    epochs = 15
    learning_rate = 0.0005
    train_model(model, train_loader, criterion, epochs=epochs, learning_rate=learning_rate)

    #Sauvegarder le modèle entraîné
    dossier_models = "pretrained_models"
    model_path = os.path.join(dossier_models, "trained_OccitanCNN.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Modèle entraîné sauvegardé dans : {model_path}")

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
