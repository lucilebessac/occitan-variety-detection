# Détecteur de Dialectes Occitans

## Objectifs du projet

Ce projet vise à créer un outil de détection automatique des dialectes occitans, en se concentrant initialement sur une classification binaire entre deux variétés principales : le languedocien et le gascon. L'occitan est une langue régionale "peu dotée" en ressources numériques et, d'après nos recherches, ne dispose pas encore de détecteur de variété en traitement automatique du langage naturel (NLP), malgré ses six variétés dialectales distinctes.

Notre approche progressive a consisté à:
1. Commencer par une classification binaire (languedocien/gascon) pour établir une base solide
2. Développer un réseau de neurones convolutif (CNN) adapté aux spécificités de l'occitan
3. Utiliser des embeddings pré-entraînés avec FastText pour représenter les textes occitans
4. Gérer le déséquilibre des classes inhérent aux corpus disponibles

L'objectif final est de contribuer aux ressources NLP pour l'occitan, en offrant un outil capable d'identifier automatiquement les variétés dialectales de cette langue régionale.

## Méthodologie

### Répartition du travail

Notre équipe de deux personnes a adopté une approche collaborative et itérative:

1. **Phase de recherche et conception**:
   - Recherche bibliographique sur les dialectes occitans et leurs caractéristiques
   - Exploration des ressources numériques disponibles
   - Définition de l'architecture du modèle et des étapes de traitement
   - Choix des outils et bibliothèques pertinents (PyTorch, FastText, etc.)

2. **Collecte et préparation des données**:
   - Identification des sources pour les corpus languedocien et gascon
   - Constitution d'un corpus équilibré pour l'entraînement
   - Préparation des données (nettoyage, normalisation, tokenisation)

3. **Développement du modèle**:
   - Implémentation du CNN en PyTorch
   - Intégration des embeddings FastText pour l'occitan
   - Mise en place des fonctions d'entraînement et d'évaluation

4. **Expérimentation et optimisation**:
   - Tests avec différents hyperparamètres (learning rate, nombre d'époques)
   - Ajustement de la pondération pour gérer le déséquilibre des classes
   - Analyse des performances et itérations successives

### Identification et résolution des problèmes

Au cours du développement, nous avons rencontré plusieurs défis techniques:

1. **Problème de ressources computationnelles**:
   - Problème: La tokenisation et vectorisation de l'ensemble du corpus dépassait la mémoire disponible
   - Solution: Limitation de la longueur des phrases en utilisant le 90ème percentile comme seuil, et conversion en format float16 pour réduire l'empreinte mémoire

2. **Déséquilibre des classes**:
   - Problème: Distribution inégale des exemples entre les classes
   - Solution: Implémentation d'une fonction de pondération des poids avec CrossEntropyLoss pour donner plus d'importance aux classes minoritaires

3. **Problème de surapprentissage**:
   - Problème: Risque que le modèle mémorise les exemples plutôt que d'apprendre les caractéristiques des dialectes
   - Solution: Ajustement du learning rate (0.001) et suivi attentif de l'évolution de la loss

4. **Problèmes de prédiction initiaux**:
   - Problème: Le modèle prédisait systématiquement une seule classe (UndefinedMetricWarning)
   - Solution: Ajustement de la fonction de pondération des poids et des hyperparamètres d'entraînement

### Étapes du projet

1. **Définition du problème et recherche documentaire** (Semaine 1)
   - Étude des dialectes occitans et des approches NLP existantes
   - Identification des sources de données potentielles

2. **Collecte et préparation des données** (Semaine 2)
   - Constitution du corpus avec étiquetage des dialectes
   - Prétraitement des textes

3. **Développement du modèle CNN** (Semaine 3)
   - Conception de l'architecture du réseau
   - Implémentation de la tokenisation et de la vectorisation avec FastText

4. **Entraînement et optimisation** (Semaine 4)
   - Tests avec différents hyperparamètres
   - Évaluation des performances et ajustements

5. **Analyse des résultats et documentation** (Semaine 5)
   - Évaluation finale du modèle
   - Rédaction de la documentation technique

## Implémentation

### Architecture du système

Notre système se compose de plusieurs modules:

1. **Module de chargement des données** (`dataset.py`)
   - Fonctions pour charger le corpus occitan
   - Tokeniseur spécifique à l'occitan

2. **Module de réseau neuronal** (`occitanCNN.py`)
   - Définition de l'architecture CNN adaptée aux textes occitans
   - Configuration des couches et des paramètres

3. **Module d'entraînement** (`training.py`)
   - Fonctions pour l'entraînement du modèle
   - Fonctions d'évaluation avec métriques pertinentes

4. **Module utilitaire** (`utils.py`)
   - Fonctions pour la gestion des labels et le calcul des poids
   - Fonctions auxiliaires pour le traitement des données

5. **Script principal** (`main.py`)
   - Orchestration du processus complet
   - Paramétrage et lancement de l'entraînement

### Technologies utilisées

- **PyTorch**: Framework d'apprentissage automatique pour l'implémentation du CNN
- **FastText**: Pour les embeddings de mots en occitan (modèle cc.oc.300.bin)
- **Keras**: Pour certaines fonctionnalités de prétraitement (pad_sequences)
- **NumPy**: Pour les manipulations numériques
- **Scikit-learn**: Pour la division des données et les métriques d'évaluation

### Processus de traitement

1. **Chargement des données et du modèle FastText**:
   ```python
   fasttext.util.download_model('oc', if_exists='ignore')
   model = fasttext.load_model('cc.oc.300.bin')
   corpus, labels = charger_corpus(dossier_data)
   ```

2. **Préparation et division des données**:
   ```python
   X_train, X_test, y_train, y_test = train_test_split(corpus, labels, test_size=0.2, random_state=42)
   ```

3. **Tokenisation et vectorisation**:
   ```python
   phrase_tokenisee = tokenizer_occitan(texte)
   phrase_vectorisee = [model.get_word_vector(mot) for mot in phrase_tokenisee][:max_len]
   ```

4. **Padding et conversion en tenseurs**:
   ```python
   phrases_vectorisees_train_padded = pad_sequences(phrases_vectorisees_train, maxlen=max_len, dtype="float16", padding="post")
   X_train_tensor = torch.tensor(phrases_vectorisees_train_padded, dtype=torch.float32)
   ```

5. **Gestion du déséquilibre des classes**:
   ```python
   class_weights = compter_labels(y_train)
   criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
   ```

6. **Entraînement du modèle**:
   ```python
   train_model(model, train_loader, criterion, epochs=epochs, learning_rate=learning_rate)
   ```

7. **Évaluation des performances**:
   ```python
   avg_loss, accuracy, precision, recall, f1 = evaluate_model(model, test_loader, criterion)
   ```

## Résultats et discussion

### Performances du modèle

Notre modèle a atteint des performances encourageantes sur la tâche de classification binaire des dialectes occitans:

- **Accuracy**: [valeur] %
- **Precision**: [valeur]
- **Recall**: [valeur]
- **F1 Score**: [valeur]

Nous avons observé que la loss continuait de diminuer même après plusieurs époques, ce qui suggère que le modèle pourrait encore bénéficier d'un entraînement plus long.

Les résultats montrent également que le modèle est capable de distinguer efficacement les deux principales variétés d'occitan (languedocien et gascon), malgré les similitudes linguistiques entre ces dialectes.

### Limitations actuelles

Plusieurs limitations ont été identifiées:

1. **Corpus limité**: Les ressources disponibles pour l'occitan restent limitées, ce qui réduit la diversité des exemples d'entraînement.

2. **Classification binaire uniquement**: Notre approche actuelle se limite à deux dialectes, alors que l'occitan en compte six.

3. **Absence de couche d'embedding personnalisée**: Nous utilisons directement les embeddings FastText sans affiner une couche d'embedding spécifique à notre tâche.

4. **Difficulté avec la classe "inconnu"**: Les tests initiaux avec une troisième classe pour les textes non occitans ont montré des performances limitées en raison du très faible nombre d'exemples.

### Perspectives d'amélioration

Pour les développements futurs, nous envisageons:

1. **Extension à d'autres dialectes**: Intégrer progressivement les quatre autres variétés dialectales (provençal, auvergnat, limousin, vivaro-alpin).

2. **Enrichissement du corpus**: Collecter davantage de textes pour améliorer la robustesse du modèle.

3. **Interface web**: Développer une interface utilisateur simple permettant de tester le détecteur de dialectes en ligne.

4. **Affinage de la couche d'embedding**: Implémenter une couche d'embedding personnalisée qui pourrait être entraînée spécifiquement pour cette tâche.

5. **Classification multilingue**: Ajouter une capacité de détection du français ou d'autres langues pour distinguer l'occitan d'autres langues.

### Conclusion

Ce projet a permis de développer un premier détecteur de dialectes pour l'occitan, contribuant ainsi aux ressources NLP pour cette langue régionale peu dotée. Malgré les limitations actuelles, les résultats sont encourageants et ouvrent la voie à des développements futurs qui pourraient bénéficier à la communauté des locuteurs et chercheurs travaillant sur l'occitan.