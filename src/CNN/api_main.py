from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from occitanCNN import OccitanCNN
from dataset import tokenizer_occitan, vectorizer_phrase, padding_liste_phrases, tensorizer_phrase, charger_fasttext # PB DE CHEMIN ?

import torch

#Initialisation de FastAPI
app = FastAPI()

class TextInput(BaseModel):
    text: str

max_len = 53
labels = ["gascon","languedocien","autre"]

# À cherger au démarrage de l'API
@app.on_event("startup")
async def load_models():
    global vec_model, cnn_model
    
    try:
        print("Chargement du modèle FastText ...")
        vec_model = charger_fasttext()
        
        print("Chargement du CNN...")
        cnn_model = OccitanCNN(fasttext_embedding_dim=300, nb_filtres=100)
        cnn_model.load_state_dict(torch.load("trained_OccitanCNN.pth"))
        cnn_model.eval()
        
        print("Modèles chargés (ouf!)")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Échec")


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

# Run l'API avec: uvicorn api:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_main:app", host="0.0.0.0", port=8000, reload=True)