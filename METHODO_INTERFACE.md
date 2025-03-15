# Interface pour Détecteur de Dialectes Occitans

## Objectif du projet

Ce projet a pour but la réalisation d'une interface web pour notre Détecteur de Dialectes Occitan. 

Nous avons précédemment développé un système de détection des variantes dialectales de l'occitan (gascon/languedocien). Ce système repose sur un modèle Text-CNN, qui a été entraîné, sauvegardé et évalué avec un corpus spécifique à ces deux variantes. Cette partie du projet consiste à créer une interface web permettant aux utilisateurs d'accéder à notre système de TAL. L'interface permet à un utilisateur d'envoyer un mot ou des phrases en occitan qui sont ensuite classifiées par le modèle CNN afin de détecter si elles appartiennent au dialecte languedocien ou gascon. 

L'objectif final de ce double projet est de contribuer aux ressources NLP pour l'occitan. Cette étape permet de rendre le modèle accessible et de faciliter son utilisation (notamment par des non-experts en programmation).

## Méthodologie

### Organisation du travail

1. **Sauvegarde et appel du modèle** :
   - Sauvegarde du modèle Text-CNN après l'entraînement en utilisant PyTorch
   - Consultation de la documentation de PyTorch pour l'exportation et le chargement du modèle
   - Adaptation des scripts pour charger et faire appel au modèle sauvegardé

2. **Choix de la technologie pour l'API** :
   - Comparaison entre FastAPI et Flask pour décider de la meilleure solution pour l'interface web

3. **Mise en place du pipeline** :
   - Développement de l'API avec FastAPI pour permettre l'envoi de phrases en occitan au modèle
   - Création du script pour l'envoi et la réception des requêtes
   - Mise en place de l'architecture du pipeline de traitement des requêtes

4. **Déploiement en ligne** :
   - Recherche d'outils pour le déploiement en ligne (PythonAnywhere, Render)
   - Tentative de déploiement du modèle, mais impossibilité d'héberger le modèle sur PythonAnywhere en raison de sa taille
   - Exploration d'autres solutions de déploiement pour garantir l'accessibilité du modèle via l'interface web
    -> Finalement il a été décidé d'abandonner le déployement en ligne (pour le moment).


### Outils utilisés

- **PyTorch** : Framework de ML qui permet de faire appel à notre Text-CNN sauvegardé
- **FastAPI** : Framework pour créer l'API web permettant l'interaction avec le modèle et la gestion des requêtes
- **HTML** : Langage de balisage utilisé pour structurer l'interface web et permettre l'affichage des résultats
- **Bootstrap** : Framework CSS pour la conception d'interfaces réactives et modernes, utilisé pour styliser l'interface web


### Pipeline

Décrite dans [README.md](README.md).


### Perspectives d'amélioration

Pour les développements futurs, nous envisageons :
- **Extension à d'autres dialectes et/ou langues**: Intégrer progressivement les quatre autres variétés dialectales (provençal, auvergnat, limousin, vivaro-alpin).
- **Déployer l'interface**: Déployer l'outil afin qu'il soit accessible en ligne.
- **Amélioration de l'interface utilisateur**: Intégrer du javascript afin de rendre l'interface dynamique et plus jolie.


### Conclusion

Ce projet nous a permis de contribuer aux ressources numériques pour cette langue régionale peu dotée.

Malgré nos limites, les résultats sont assez encourageants et nous pensons qu’en effectuant le travail nécessaire pour la collecte de données pour constituer des corpus, un détecteur prenant en compte toutes les variétés pourrait être développé.


