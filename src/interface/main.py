from fastapi import FastAPI

from CNN.occitanCNN import OccitanCNN ## PB DE CHEMIN À RÉSOUDRE
from CNN.dataset import tokenizer_occitan, vectorizer_phrase, padding_liste_phrases, tensorizer_phrase, charger_fasttext # PB DE CHEMIN À RÉSOUDRE
from post_input import TextInput
import torch

#Initialisation de FastAPI
app = FastAPI()


#Cherche si gpu ou cpu -> optimise 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Importer le modèle pretrained
cnn_model = OccitanCNN().to(device)
cnn_model.load_state_dict(torch.load("trained_OccitanCNN.pth")) # PATH A CHANGER

vec_model = charger_fasttext()

#Permet de tester si l’API fonctionne.
@app.get("/")
def root():
    return {"message": "Bonjorn ! Adishatz ! Bonjour !"}


#Requête de l'utilisateur (POST)
@app.post("/prediction/")
def prediction(data: TextInput): # TextPost est un objet qui contient un attribut text / il faut le définir

    # transformation dans un format qui convient à notre modèle
    input_tokenise = tokenizer_occitan(data.text)
    input_vectorise = vectorizer_phrase(input_tokenise, vec_model, 56) # max_len = 56, à récupérer automatiquement / vérifier
    input_vectorise_padded = padding_liste_phrases(input_vectorise, 56) # max_len = 56, à récupérer automatiquement / vérifier
    input_tensor = tensorizer_phrase(input_vectorise_padded, torch.float32)

    # prédiction
    prediction = cnn_model(input_tensor)

    return {"text": data.text, "prediction": prediction}



