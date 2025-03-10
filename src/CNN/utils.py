from collections import Counter
import torch
import os
import csv

def compter_labels(labels, nbr_classes) : #obtenir des poids selon la proportion de la classe
    
    total_labels = len(labels)
    class_counts = Counter(labels)
    weights = [total_labels / class_counts.get(i, 1) for i in range(nbr_classes)]

    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum()
    
    return weights

def save_results(epochs, learning_rate, avg_loss, accuracy, precision, recall, f1, classif_report): 
    
    dossier_resultats = "../results"
    os.makedirs(dossier_resultats, exist_ok=True)
    nom_fichier = f"epoch={epochs}_lr={learning_rate}.csv"
    chemin_fichier = os.path.join(dossier_resultats, nom_fichier)

    with open(chemin_fichier, mode="w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["Epoch", "Learning Rate", "Average Loss", "Accuracy", "Precision", "Recall", "F1 Score", "Classification Report"])
        writer.writerow([epochs, learning_rate, avg_loss, accuracy, precision, recall, f1, classif_report])
