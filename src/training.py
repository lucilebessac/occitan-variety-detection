import torch
import torch.nn as nn
import torch.optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def train_model(model, train_loader, criterion, epochs, learning_rate):
    """
    criterion : fait appel à CrossEntropyLoss dans le main pour appliquer les poids liés au déséquilibre de classes
    epochs=15 pour le premier test
    lr = 0.03 pour le premier test
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Version de la Descente Gradient Stochastique 
    #il est déjà implémenté sous la forme d'un objet auquel on passe les paramètres à optimiser (les poids)

    loss_history = []
    for epoch in range(epochs):
        model.train() # Passer en mode entrainement
        epoch_loss = 0.0

        for input, label in train_loader : # Parcourt chaque batch de train
            input = input.float()
            outputs = model(input) # Passe les données dans le modèle, outputs = tensor contenant les scores de proba pour chaque classe
            loss = criterion(outputs, label) # Calcul de la loss : comparaison pred du modèle (outputs) avec les étiquettes réelles (label)
            loss.backward() # Rétropropagation 
            optimizer.step() # On applique un pas de l'algo de descente de gradient, c'est ici qu'on modifie les poids
            optimizer.zero_grad() # remettre les gradients des paramètres à zéro sinon ils s'accumulent quand on appelle `backward`

            epoch_loss += loss.item() 
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")


def evaluate_model(model, test_loader, criterion):
    
    model.eval() # Passer en mode évaluation
    total_loss = 0.0
    total_pred = []
    total_exemples = []

    with torch.no_grad(): #comme c'est l'éval, pas besoin du calcul de gradient -> réduction du coût mémoire
        for input, label in test_loader :  # Parcourt chaque batch de test
            input = input.float()
            outputs = model(input) #Passe les données dans le modèle, outputs = tensor contenant les scores de proba pour chaque classe
            loss = criterion(outputs, label) #Calcul de la loss : comparaison pred du modèle (outputs) avec les étiquettes réelles (label)
            total_loss += loss.item() #ajout loss de ce batch à la perte totale
        
            # Calcul des prédictions
            valeurs_max, predicted = torch.max(outputs, 1) #torch.max(tensor, dim)dim=1 car on cherche sur l'axe des colonnes pour trouver pour chaque ligne (phrase), quelle est la classe la plus probable
            
            total_exemples.extend(label.cpu().numpy()) # Stocke les labels réels  #.cpu c'est pour repasser en cpu car numpy ne manipule pas les trucs sur gpu
            total_pred.extend(predicted.cpu().numpy()) # Stocke les prédictions

        avg_loss = total_loss / len(test_loader)

        # Calcul des métriques
        classif_report = classification_report(total_exemples, total_pred)
        accuracy = accuracy_score(total_exemples, total_pred)
        precision = precision_score(total_exemples, total_pred, average='weighted')
        recall = recall_score(total_exemples, total_pred, average='weighted')
        f1 = f1_score(total_exemples, total_pred, average='weighted')

    return avg_loss, accuracy, precision, recall, f1, classif_report