import torch
import torch.nn as nn

### C'est un script brouillon encore ###
#sur internet, on fait généralement une classe mais le prof lui, il a fait une fonction donc jsp à voir

## COUCHE EMBEDDING
#faut créer une matrice de notre vocabulaire (sur fast text y'a pas la méthode simple comme avec w2vec gensim :/)
#on veut un tenseur : (taille_vocab, embedding_dim), nous, embedding_dim=300
#pretrained_embedding = torch.
embedding_layer = nn.Embedding.from_pretrained(pretrained_embedding, freeze=False, padding_idx=0)

## COUCHES CONVOLUTIVES
conv1 = nn.Conv1d(out_channels=64, kernel_size=2, stride=1, padding=1) # Couche 1 : filtre taille 2    
relu1 = nn.ReLU()                                          
conv2 = nn.Conv1d(out_channels=64, kernel_size=3, stride=1, padding=1) # Couche 2 : filtre taille 3
relu2 = nn.ReLU()
conv3 = nn.Conv1d(out_channels=64, kernel_size=4, stride=1, padding=2) # Couche 3 : taille filtre taille 4
relu3 = nn.ReLU()

    ## Activation ReLU jsp trop si c'est comme ça :) ?

## COUCHE MAX POOLING
pool = nn.MaxPool1d(kernel_size=2 , stride=1) #petite réduction mais conserve plus d'informations, à test avec stride=2 aussi 

## SOFTMAX



