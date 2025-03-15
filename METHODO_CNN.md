# Détecteur de Dialectes Occitans

## Objectifs du projet

L'occitan est une langue régionale "peu dotée" en ressources numériques et qui, d'après nos recherches et les ressources listées sur [lafarganumerica.eu](https://lafarganumerica.eu/etat-des-lieux/les-grands-domaines/215-detection-de-langue-et-de-variete), ne dispose pas encore de détecteur automatique de variété pour ses six variétés dialectales. Ce projet vise à créer un outil de détection automatique des 6 dialectes occitans.


Nos objectifs idéaux sont de:
1. Développer un classifieur à 7 classes : les 6 variété de l’occitan + une classe “autre” pour tout texte qui n’est pas en occitan ou qui n’est pas détecté avec un certain seuil de confiance
2. Apprendre nos propres embeddings pour chaque dialecte
3. Développer un réseau de neurones adapté à l'occitan afin de mieux classifier 


L'objectif final est de contribuer aux ressources NLP pour l'occitan, en développant un outil utile et non encore existant.


Notre méthodologie a été de commencer plus petit afin d'atteindre nos objectifs pas à pas.
Nous avons donc commencé avec :
1. Classification binaire (languedocien/gascon) pour avoir une base solide et car ce sont les 2 variantes les plus répandues et avec le plus de ressources (corpus) numériques
2. Utilisation des embeddings pré-entraînés avec FastText pour représenter les textes occitans


## Méthodologie

### Organisation du travail

1. **Recherche et conception**:
   - Recherche sur les dialectes occitans et [leurs caractéristiques](https://webetab.ac-bordeaux.fr/Primaire/64/oloron/themes/file/occitan/Cartes/carte_variantes_c.pdf)
   - Recherches sur les [ressources numériques (corpus et outils) disponibles](https://lafarganumerica.eu/a-disposition/inventaire-des-ressources-numeriques)
   - [État de l’art de la détection de dialectes en occitan](https://www.academia.edu/28154017/Reconnaissance_automatique_des_dialectes_occitans_%C3%A0_l%C3%A9crit)
   - Définition de l'architecture du modèle et des étapes de traitement
   - Choix des outils et bibliothèques pertinents (PyTorch, FastText, etc.)
2. **Corpus**:
   - Constitution d'un corpus pour l'entraînement à partir de [ressources disponibles sur la farga numerica](https://lafarganumerica.eu/a-disposition/inventaire-des-ressources-numeriques ) et complétion avec des [ressources en ligne](https://www.bilinguisme-occitan.org/comptines)
   - Préparation des données (nettoyage, normalisation, tokenisation)
3. **Développement du modèle**:
   - Embeddings FastText pour l'occitan
   - Construction du CNN en PyTorch
   - Fonctions d'entraînement et d'évaluation
4. **Expérimentation et optimisation**:
   - Tests avec différents hyperparamètres (learning rate, nombre d’epoch)
   - Ajustement de la pondération pour gérer le déséquilibre des classes
   - Analyse des performances et itérations

### Outils utilisés
- **PyTorch**: Framework de ML qui permet d’implémenter le CNN
- **FastText**: Pour les embeddings de mots en occitan (modèle cc.oc.300.bin)
- **Tokenizer**: Tokenization for Occitan (Gascon and Lengadocian) par Marianne Vergez-Couret et Miletic Aleksandra
- **Keras**: Pour certaines fonctionnalités de prétraitement (pad_sequences)
- **NumPy**: Pour les manipulations numériques / matricielles
- **Scikit-learn**: Pour la division des données et les métriques d'évaluation

### Architecture du réseau neuronal
- **Pourquoi un CNN ?**:
   - D’après nos recherches, les CNN nécessitent souvent moins de données d'entraînement que d’autres architectures, ce qui est particulièrement intéressant dans notre contexte avec une langue peu dotée.
   - Les CNN semblent détecter des motifs indépendamment de leur position, ce qui est adapté ici puisque les principales différences entre dialectes de l’occitan sont lexicales et orthographiques et non syntaxiques. Pas besoin donc d’utiliser un RNN qui prendrait en compte l’ordre des tokens.
   - De plus, les CNN son plus rapides à entraîner ce qui est une différence non négligeable.
- **Couches convolutives : Pourquoi ces 3 tailles de filtres ?** : 
  - Pour introduire des granularités différentes (Bi-grams, tri_grams, quadri-grams)
- **Couche Pooling : Pourquoi Max pooling ?** : 
   - Sélectionner la valeur maximum afin de capturer les caractéristiques les plus importantes tout en réduisant la dimensionnalité.
- **Entrainement : Pourquoi ces hyperparamètres ?** : 
   - Learning rate : 0.0005
   - Epoch : 15  
   Après plusieurs tests, nous avons obtenu les meilleurs performances avec les hyperparamètres ci-dessus.

### Pipeline

Décrite dans [README.md](README.md)

## Résultats et discussion

### Performances
- Epoch : 15 
- LR : 0.0005 
- Accuracy : 0.9747
- F1 : 0.9698

### Limites

- **Corpus limité**: Les ressources pour l'occitan sont limitées, même pour les 2 dialectes les mieux dotés.
- **Données parfois peu qualitatives** : Orthographe pas toujours fixée, mélanges de dialectes, mélanges de langues, phrases mal découpées …
- **Classification binaire uniquement**: Nous nous limitons pour l’instant à 2 dialectes au lieu de 6.
- **Absence de couche d'embedding personnalisée**: Nous utilisons directement les embeddings FastText sans apprendre une couche d'embedding spécifique à notre tâche.
- **Difficulté avec la classe "autre"**: Nous aimerions que notre classifieur intègre une classe “autre” ou “inconnu” au cas où on lui donne en input quelque chose qui n’est ni du gascon ni du languedocien. Nous avons commencer à l'integrer à notre corpus comme ébauche.
- **Ressources computationnelles** : Essai d'adapter le script pour convenir à nos ressources computationnelles mais différents messages d'erreurs. Nous avons été obligées de nous organiser car seule une personne du groupe avait les ressources pour lancer les scripts complets d'entrainement/évaluation (avec des temps très longs).


### Perspectives d'amélioration
Pour les développements futurs, nous envisageons:
- **Extension à d'autres dialectes et/ou langues**: Intégrer progressivement les quatre autres variétés dialectales (provençal, auvergnat, limousin, vivaro-alpin).
- **Enrichissement du corpus**: Collecter davantage de textes et de meilleure qualité.
- **Détection du “code-switching”** : l’idéal serait de réussir à détecter les mélanges de langues ou de dialectes.
- **Embedding**: Implémenter des embeddings appris.

### Conclusion
Ce projet nous a permis de développer une première version d’un détecteur neuronal de dialectes pour l’occitan et de contribuer aux ressources numériques pour cette langue régionale peu dotée.


Malgré nos limites, les résultats sont assez encourageants et nous pensons qu’en effectuant le travail nécessaire pour la collecte de données pour constituer des corpus, un détecteur prenant en compte toutes les variétés pourrait être développé.

Il serait intéressant de comparer cette approche à une approche moins coûteuse en puissance de calcul comme une technique hybride à base de règles. Il ne semble peut-être pas nécessaire, pour une simple tâche de classification et alors que les différences entre les dialectes sont lexicales et orthographiques, d’utiliser un réseau de neurones. 
