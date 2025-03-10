from fastapi import FastAPI
import torch

#Initialisation de FastAPI
app = FastAPI()


#Cherche si gpu ou cpu -> optimise 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





#Permet de tester si l’API fonctionne.
@app.get("/")
def root():
    return {"message": "Bonjorn ! Adishatz ! Bonjour !"}


#Requête de l'utilisateur (POST)
#@app.post("/prediction/")

    #transformation dans un format qui convient à notre modèle
        #tokeniser
        #vectoriser
        #padding

    #plus ou moins comme la fonction evaluate_model



#return {"text": data.text, "prediction": label}



