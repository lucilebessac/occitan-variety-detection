import torch
from dataset import tokenizer_occitan, vectorizer_phrase, padding_liste_phrases, tensorizer_phrase, charger_fasttext
from occitanCNN import OccitanCNN
import numpy as np

phrase_test = "au patonet que l'agrada de cantar de dançar e d'escotar musica que's ditz que poiré tocar d'un instrument."

model = charger_fasttext()

phrase_tokenisee = tokenizer_occitan(phrase_test)
print("Taille de la phrase_tokenisee:", len(phrase_tokenisee))

max_len = 53
phrase_vectorisee = vectorizer_phrase(phrase_tokenisee, model, max_len)
print("phrase_vectorisee shape:", np.array(phrase_vectorisee).shape)

# print("phrase_vectorisee:", phrase_vectorisee)

phrase_vectorisee_padded = padding_liste_phrases(phrase_vectorisee, max_len)
print("phrase_vectorisee_padded shape:", np.array(phrase_vectorisee_padded).shape)
# print("phrase_vectorisee_padded:", phrase_vectorisee_padded)

phrase_tensor = tensorizer_phrase(phrase_vectorisee_padded, torch.float32)
print("phrase_tensor 1:", phrase_tensor.shape)

# utiliser le trained_OccitanCNN.pth pour charger le modèle et prédire la classe de la phrase
# charger le modèle
cnn_model = OccitanCNN(fasttext_embedding_dim=300, nb_filtres=100)
cnn_model.load_state_dict(torch.load("trained_OccitanCNN.pth"))
cnn_model.eval()


# prédiction
with torch.no_grad():
    # phrase_tensor = phrase_tensor.permute(0, 2, 1)
    prediction = cnn_model(phrase_tensor)
    classe_predite = torch.argmax(prediction, dim=1).item()
    print("classe_predite:", classe_predite)
    print("classe_predite_label:", ["gascon","languedocien","autre"][classe_predite])

