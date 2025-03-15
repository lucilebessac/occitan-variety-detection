# occitan-variety-detection
A tool for detecting Langadocian and Gascon Occitan varieties using a Convolutive Neural Network

## Description
Ce projet est un **classifieur de variété de textes en langue occitane** utilisant un réseau de neurones convolutif (CNN).

> Le modèle analyse les textes occitans et les catégorise selon leur variété : occitan languedocien ou occitan gascon.

L'application est accessible via une **API REST** avec **FastAPI** qui s'exécute localement et permet d'envoyer des phrases en occitan et d'obtenir leur classification automatique via une interface web.

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
   
## Utilisation
**1. Lancez l'API :**
```
uvicorn app:app --reload
```

**2. Une fois l'API lancée, accédez-y à l'adresse**
```
http://127.0.0.1:8000/
```
**3. Dans le champ dédié, entrez une phrase en occitan gascon ou  languedocien et appuyez sur "Analyser".**

Le dialecte de la phrase apparaît à l'écran.

Vous pouvez utiliser [le traducteur automatique revirada](https://revirada.eu/index) pour traduire des phrases du français vers l'occitan.

---

#### 📌 Exemples de phrases en occitan languedocien

> **Una femna es dins l'ostal**  

> **Lo pichon gat es mòrt.**

> **E lo Solelh l’escaudurèt, la Luna esclairèt lo camin e lo Corbàs l’ajudèt a amassar la farina.**  
> *— Andrieu Lagarda (2019) ; L’aucelon de las sèt colors ; ed. : Letras d'Òc*

> **Agaitèri a l'entorn de ieu per me rementar la topografia del lòc.**  
> *— Eliana Pignol (2020) ; « Ortolejar pòt èsser un flum longarut e intranquil » in Coups de théâtre au jardin partagé ; ed. : Le lecteur du Val*

> **Desvolopem d'espleches de TAL per las lengas regionalas !**
---

#### 📌 Exemples de phrases en occitan gascon

> **Ua hemna es dins l'ostau.**  

> **Lo gat petit qu'ei mòrt.**

> **Aquí un cotèth d'aur entà't dehéner e entà copar l'èrba blua, l'èrba qui canta nueit e dia, l'èrba qui brigalha lo hèr.**  
> *— Joan-Francés Bladé (2014) ; Lo Rei de las Agraulas ; ed. : Letras d'Òc*

> **Los mèstes que't vòlen escanar tà't har rostir.**  
> *— Andrieu Lagarda (2015) ; L’ostau deus Lops ; ed. : Letras d'Òc*

> **Desvolopem atrunas de TAL entà las lengas regionaus !**

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

> La pipeline de création, entraînement et évaluation du CNN est détaillée dans [README_CNN.md](README_CNN.md)

> Notre méthodologie pour la création, l'entraînement et l'évaluation du CNN est détaillée dans [METHODO.md](METHODO.md)

> Notre méthodologie pour la création de l'API est détaillée dans [METHODO_INTERFACE.md](METHODO_INTERFACE.md)

> La composition de notre corpus d'entraînement et de test est décrite dans [README_CORPUS.md](README_CORPUS.md)