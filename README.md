P7 - Déployez un modèle dans le cloud

Objectif du projet

Ce projet vise à déployer un modèle de machine learning dans une infrastructure cloud scalable, en utilisant AWS comme plateforme principale. L’objectif est de mettre en place une chaîne de traitement Big Data capable de gérer des volumes de données croissants, tout en respectant les contraintes de performance, de coût et de conformité (RGPD). Le modèle déployé s’appuie sur un cas d’usage financier (prédiction de défaut de paiement), basé sur les données de Home Credit.

Contenu du repository:

.github/workflows : Configuration des workflows CI/CD pour automatiser les tests et le déploiement.
.gitignore : Exclusion des fichiers non pertinents pour le versionnement.
HomeCredit_columns_description_translated.csv : Description traduite des colonnes du jeu de données Home Credit.
app.py : Script principal pour l’application API déployée.
dashboard.py : Script pour générer un tableau de bord interactif.
expected_value.pkl : Fichier contenant les valeurs attendues pour validation.
mon_best_modele_entraine_LightGBM.pkl : Modèle LightGBM entraîné.
request_app.py : Script pour effectuer des requêtes vers l’API.
requirements.txt : Dépendances Python nécessaires.
shap_values.pkl : Valeurs SHAP pour l’interprétabilité du modèle.
top_50_train.csv : Jeu de données d’entraînement réduit (50 premières lignes).
top_50_vraiVal_X_train.csv : Données de validation correspondantes.
unittest_app.py : Tests unitaires pour valider l’application.

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
