from fastapi import FastAPI

from occitanCNN import OccitanCNN
from dataset import tokenizer_occitan, vectorizer_phrase, padding_liste_phrases, tensorizer_phrase, charger_fasttext # PB DE CHEMIN À RÉSOUDRE
import torch
from pydantic import BaseModel


#Initialisation de FastAPI
app = FastAPI()

max_len = 53
labels = ["languedocien","gascon","autre"]

#Cherche si gpu ou cpu -> optimise 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Importer le modèle pretrained
cnn_model = OccitanCNN(fasttext_embedding_dim=300, nb_filtres=100)
cnn_model.load_state_dict(torch.load("model.pth")) # PATH A CHANGER
cnn_model.eval()

vec_model = charger_fasttext()

class TextInput(BaseModel):
    text: str

#Permet de tester si l’API fonctionne.
@app.get("/")
def root():
    return {"message": "Bonjorn ! Adishatz ! Bonjour !"}


#Requête de l'utilisateur (POST)
@app.post("/prediction/")
async def prediction(data: TextInput): # TextPost est un objet qui contient un attribut text / il faut le définir

    # transformation dans un format qui convient à notre modèle
    input_tokenise = tokenizer_occitan(data.text)
    input_vectorise = vectorizer_phrase(input_tokenise, vec_model, max_len) # max_len = 53
    input_vectorise_padded = padding_liste_phrases(input_vectorise, max_len) # max_len = 53
    input_tensor = tensorizer_phrase(input_vectorise_padded, torch.float32)

    # prédiction
    with torch.no_grad():
        prediction = cnn_model(input_tensor)
        classe_predite = torch.argmax(prediction, dim=1).item()

    return {"text": data.text, "prediction": labels[classe_predite]}



