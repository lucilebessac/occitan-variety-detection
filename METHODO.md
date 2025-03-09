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

Notre équipe de deux personnes a adopté une approche collaborative et itérative:

1. **Recherche et conception**:
   - Recherche sur les dialectes occitans et [leurs caractéristiques](https://webetab.ac-bordeaux.fr/Primaire/64/oloron/themes/file/occitan/Cartes/carte_variantes_c.pdf)
   - Recherches sur les [ressources numériques (corpus et outils) disponibles](https://lafarganumerica.eu/a-disposition/inventaire-des-ressources-numeriques)
   - [État de l’art de la détection de dialectes en occitan](https://www.academia.edu/28154017/Reconnaissance_automatique_des_dialectes_occitans_%C3%A0_l%C3%A9crit)
   - Définition de l'architecture du modèle et des étapes de traitement
   - Choix des outils et bibliothèques pertinents (PyTorch, FastText, etc.)