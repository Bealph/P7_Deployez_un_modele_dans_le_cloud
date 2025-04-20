Projet "Déployez un modèle dans le cloud"

Introduction

Ce projet a été réalisé dans le cadre du parcours de formation en Data Science avec OpenClassrooms. Il vise à développer un outil de scoring de crédit pour la société financière "Prêt à dépenser". Cet outil calcule la probabilité qu’un client rembourse son crédit et classifie la demande en crédit accordé ou refusé. Un dashboard interactif est également développé pour permettre aux chargés de relation client d’expliquer les décisions d’octroi de crédit de manière transparente et de fournir aux clients un accès facile à leurs informations personnelles.

Objectifs du projet :

- Construire un modèle de scoring basé sur des données variées pour prédire la probabilité de faillite d’un client.
- Développer un dashboard interactif pour visualiser les scores et les informations des clients.
- Mettre en production le modèle via une API.
- Assurer la transparence et l’interprétabilité du modèle grâce à des outils comme SHAP.
- Prendre en compte le déséquilibre des classes et le coût métier dans l’élaboration du modèle.

Structure du repository :

- .github/workflows : Configurations pour les workflows GitHub (intégration et déploiement continu).
- .gitignore : Fichier pour ignorer certains fichiers lors des commits.
- HomeCredit_columns_description_translated.csv : Description traduite des colonnes des données.
- app.py : Fichier principal pour l’API de prédiction.
- dashboard.py : Fichier pour le dashboard interactif.
- expected_value.pkl : Fichier contenant les valeurs attendues pour le modèle.
- image_app.jpeg : Image utilisée dans l’application.
- mon_best_modele_entraine_LightGBM.pkl : Modèle entraîné avec LightGBM.
- request_app.py : Script pour envoyer des requêtes à l’API.
- requirements.txt : Liste des dépendances du projet.
- shap_values.pkl : Valeurs SHAP pour l’interprétabilité du modèle.
- top_50_train.csv : Données d’entraînement avec les 50 meilleures features.
- top_50_vraiVal_X_train.csv : Données d’entraînement pour la validation avec les 50 meilleures features.
- unittest_app.py : Tests unitaires pour l’API.

Livrables :

- Application de dashboard interactif : Déployée sur le cloud, elle permet de visualiser les scores et les informations clients.
- API de prédiction : Déployée sur le cloud pour fournir des prédictions en temps réel.
- Notebook de modélisation : Contient le prétraitement, l’entraînement et l’évaluation du modèle avec tracking via MLFlow.
- Code source : Inclut le code du dashboard (dashboard.py) et de l’API (app.py).
- Tableau HTML d’analyse de data drift : Réalisé avec Evidently pour détecter les dérives de données.
- Note méthodologique : Décrit la démarche d’élaboration du modèle, le traitement des déséquilibres, et l’analyse de data drift.

Conclusion :

Ce projet illustre la mise en œuvre d’un modèle de scoring de crédit avec une API et un dashboard interactif, tout en priorisant la transparence et l’interprétabilité. N’hésitez pas à explorer le repository et à contribuer si vous le souhaitez !
