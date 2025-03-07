from collections import Counter
import torch

def compter_labels(labels) : #obtenir des poids selon la proportion de la classe
    
    total_labels = len(labels)
    class_counts = Counter(labels)
    weights = [total_labels / class_counts[i] for i in range(len(class_counts))]

    return torch.tensor(weights, dtype=torch.float16)