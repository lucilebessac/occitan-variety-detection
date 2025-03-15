from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import HTMLResponse
from occitanCNN import OccitanCNN
from dataset import tokenizer_occitan, vectorizer_phrase, padding_liste_phrases, tensorizer_phrase, charger_fasttext # PB DE CHEMIN ?

import torch

#Initialisation de FastAPI
app = FastAPI()

max_len = 53
labels = ["gascon","languedocien","autre"]


# Charger les modèles
@app.on_event("startup")
async def load_models():
    global vec_model, cnn_model
    try:
        print("Chargement du modèle FastText ...")
        vec_model = charger_fasttext()

        print("Chargement du CNN...")
        cnn_model = OccitanCNN(fasttext_embedding_dim=300, nb_filtres=100)
        cnn_model.load_state_dict(torch.load("pretrained_models/trained_OccitanCNN.pth"))
        cnn_model.eval()

        print("Modèles chargés (ouf!)")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Échec")

# Page d'accueil avec bootstrap
@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Détecteur de variétés pour l'Occitan</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body class="container mt-5">
        <div class="text-center">
            <h1 class="mb-4">Bonjorn ! Adishatz ! Bonjour !</h1>
            <p class="lead">
                Bienvenue sur ce détecteur de variétés de l'Occitan.
            </p>
            <p>Entrez une phrase en Occitan pour savoir si elle est en dialecte languedocien ou gascon.</p>
            <p class="text-danger">Attention : les autres dialectes (provençal, auvergnat, vivaro-alpin, limousin) ne sont pas (encore) pris en charge.</p>
        </div>

        <div class="row justify-content-center mt-4">
            <div class="col-md-8">
                <form action="/prediction" method="post">
                    <div class="input-group mb-3">
                        <input type="text" name="text" class="form-control" placeholder="Entrez une phrase en Occitan" required>
                        <button class="btn btn-primary" type="submit">Analyser</button>
                    </div>
                </form>
            </div>
        </div>
    </body>
    </html>
    """)

# Requête de l'utilisateur et prédiction
@app.post("/prediction", response_class=HTMLResponse)
async def prediction(text: str = Form(...)):
    # Transformation dans un format que le modèle peut traiter
    input_tokenise = tokenizer_occitan(text)
    input_vectorise = vectorizer_phrase(input_tokenise, vec_model, max_len)
    input_vectorise_padded = padding_liste_phrases([input_vectorise], max_len)
    input_tensor = tensorizer_phrase(input_vectorise_padded, torch.float32)

    # Prédiction
    with torch.no_grad():
        prediction = cnn_model(input_tensor)
        classe_predite = torch.argmax(prediction, dim=1).item()

    # Renvoie une page HTML avec les résultats
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Résultat de l'analyse</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body class="container mt-5">
        <div class="text-center">
            <h1 class="mb-4">Bonjorn ! Adishatz ! Bonjour !</h1>
            <p class="lead">
                Voici le résultat de l'analyse du texte que vous avez soumis :
            </p>
            <div class="card">
                <div class="card-header bg-success text-white">
                    Résultat de l'analyse
                </div>
                <div class="card-body">
                    <p><strong>Texte analysé :</strong> {text}</p>
                    <p><strong>Dialecte détecté :</strong> {labels[classe_predite]}</p>
                </div>
            </div>
            <a href="/" class="btn btn-primary mt-4">Retour à l'accueil</a>
        </div>
    </body>
    </html>
    """)
