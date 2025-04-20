P7 - Déployez un modèle dans le cloud

Objectif du projet

Développé dans le cadre d'un projet AgriTech, il vise à créer une application mobile permettant aux utilisateurs de photographier un fruit et d'obtenir des informations sur celui-ci. L'objectif est double : sensibiliser le grand public à la biodiversité des fruits tout en posant les bases d'une architecture Big Data scalable pour la classification d'images. Ce repository met en œuvre une première chaîne de traitement des données avec AWS EMR et PySpark, préparant le terrain pour une montée en volume de données.

Contenu du repository:

.github/workflows : Configuration des workflows GitHub pour l'intégration et le déploiement continus (CI/CD).
.gitignore : Fichiers et dossiers exclus du contrôle de version Git.
app.py : Script principal pour le traitement des données sur AWS EMR avec PySpark.
dashboard.py : Script de visualisation des résultats.
request_app.py : Script pour les requêtes à l'application.
requirements.txt : Liste des dépendances Python.
top_50_train.csv, top_50_vraiVal_X_train.csv : Datasets pour l'entraînement et la validation.
unittest_app.py : Tests unitaires pour valider l'application.

Architecture cloud :

L’architecture repose sur les services AWS :

S3 : Stockage des données brutes et des fichiers générés (ex. matrices PCA).
EMR : Traitement distribué avec PySpark pour paralléliser les calculs.
IAM : Gestion des accès sécurisés.

Les serveurs sont hébergés en Europe pour respecter le RGPD.

Traitement des données :

Un script PySpark (non inclus dans ce repository mais prévu dans un notebook cloud) implémente :

Broadcast : Diffusion des poids du modèle LightGBM sur les nœuds EMR.

PCA : Réduction de dimension pour optimiser le traitement des données.

Gestion des coûts :

L’instance EMR est activée uniquement pour les tests et démos, minimisant ainsi les coûts.

Livrables :

Notebook PySpark exécutable sur AWS EMR.
Fichiers CSV stockés sur S3 (données initiales et résultats PCA).
Support de présentation pour la soutenance.
