from collections import Counter
import torch

def compter_labels(labels, nbr_classes) : #obtenir des poids selon la proportion de la classe
    
    total_labels = len(labels)
    class_counts = Counter(labels)
    weights = [total_labels / class_counts.get(i, 1) for i in range(len(class_counts))]

    weights = torch.tensor(weights, dtype=torch.float32)
    print(f" Avant normalisation : {weights}") #test

    weights = weights / weights.sum()
    print(f" Après normalisation : {weights}") #test

    return weights