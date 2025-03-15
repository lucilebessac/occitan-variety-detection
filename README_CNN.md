# occitan-variety-detection
A tool for detecting Langadocian and Gascon Occitan varieties using a Convolutive Neural Network

## Description
Ce projet est un classifieur de variété de textes en langue occitane utilisant un réseau de neurones convolutif (CNN). Le modèle analyse les textes occitans et les catégorise selon leur variété : occitan languedocien ou occitan gascon.

## Prérequis
- Python 3.6+
- Les dépendances listées dans `requirements.txt`

## Installation
1. Clonez ce dépôt en HTTPS :
   ```
   git clone https://github.com/lucilebessac/occitan-variety-detection.git
   ```
   Ou en SSH :
      ```
   git clone git@github.com:lucilebessac/occitan-variety-detection.git
   ```

2. Installez les dépendances listées dans `requirements.txt`
   ```
   pip install -r requirements.txt
   ```
   

## Configuration et corpus
Avant de lancer le programme, adaptez le chemin vers vos données si vous souhaitez le réentrainer sur de nouvelles données, dans le programme principal `main.py`:
```python
# line 15
dossier_data = "../data"
```

NB : le corpus doit contenir un ou plusieurs fichiers `.csv` de 2 colonnes ayant comme séparateur `§`.

La colonne 1 contient une phrase en occitan et la colonne 2 contient le label :
- 0 pour le gascon
- 1 pour le languedocien
- 2 pour autre

Des outils pour formatter votre corpus sont disponibles dans le dossier `./src/preprocess_corpus`

> La composition de notre corpus d'entraînement et de test est décrite dans [README_CORPUS.md](README_CORPUS.md)

## Utilisation
Lancez le programme principal :
```
python main.py
```

## Structure du Projet
- `src/main.py` : Programme principal pour l'entraînement du CNN.
- `src/dataset.py` : Fonctions pour charger et tokenizer le corpus occitan pour l'entraînement du CNN
- `src/occitanCNN.py` : Définition du CNN
- `src/utils.py` : Fonctions outils (équilibrer les poids selon les classes et sauvegarde des résultats)
- `src/training.py` : Fonctions d'entraînement et d'évaluation
- `src/app.py` : api
- `scripts_de_debuggage/` : outils de debuggage
- `results/` : résultats de l'entraînement.
- `pretrained_models/` : CNN et vecteurs pré entrainés
- `preprocess_corpus/` : scripts pour formater le corpus d'entraînement

## Pipeline
Le programme suit les étapes suivantes :
1. Charger le modèle FastText pour l'occitan
2. Charger et préparer le corpus
3. Diviser les données en ensembles d'entraînement et de test
4. Tokeniser et vectoriser les textes avec FastText
5. Convertir en tensors exploitables par Pytorch
6. Faire des batches pour l'entrainement
7. Gérer les déséquilibres de classe en ajustant les poids pendant l'entrainement avec une CrossEntropyLoss
8. Créer le CNN
   1. 3 couches convolutives avec différents kernels
   2. Max pooling
   3. Dropout
   4. Softmax
9.  Entrainer le CNN
10. Évaluler le CNN (précision, rappel, F1)

> Notre méthodologie de travail est détaillée dans [METHODO.md](METHODO.md)

## Paramètres
- Taille de batch : 16
- Nombre d'epoch : 15
- Leraning rate : 0.0005

## Résultats
À la fin de l'exécution, le programme affiche les métriques :
- Loss moyenne
- Rapport de classification
- Accuracy
- Precision
- Rappel
- F-mesure