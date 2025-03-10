import torch
import torch.nn as nn

class OccitanCNN(nn.Module):
    def __init__(self, fasttext_embedding_dim=300, nb_filtres=100):
        super(OccitanCNN, self).__init__()

    #Couches convolutives

        # Couches convolutives avec différentes tailles de filtres (kernels) : Bi-grams, tri_grams etc
        self.conv1 = nn.Conv1d(in_channels=fasttext_embedding_dim, out_channels=nb_filtres, kernel_size=2) 
        self.conv2 = nn.Conv1d(in_channels=fasttext_embedding_dim, out_channels=nb_filtres, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=fasttext_embedding_dim, out_channels=nb_filtres, kernel_size=4)

        # Dropout pour éviter l'overfitting . On désactive des neurones aléatoirement avec une proba choisie : ici 0.5
        self.dropout = nn.Dropout(0.5)

        # Couche "fully connected layer" pour la classification
        self.fc = nn.Linear(nb_filtres*3, 3) # *3 car on a 3 couches, puis 3 car on a 3 classes ? Enfin là dans les labels on en a que 2 mais on en veut 3 à la fin donc jsp

    def forward(self, x):
        # x est un tensor de taille (batch_size, max_len, fasttext_embedding_dim)
        # On veut un tensor de taille (batch_size, nb_filtres*3) à la fin
        x = x.permute(0, 2, 1) # On permute les dimensions pour que la taille des embeddings soit en 2e position. C'est pour que ce soit compatible avec PyTorch
        
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
        x = self.dropout(x)

    # Couche fully connected
        # On applique la couche fully connected
        x = self.fc(x)

    #Couche Softmax
        # On applique la fonction softmax pour obtenir des probabilités
        x = nn.functional.softmax(x, dim=1)

        return x



