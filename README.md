# OC Projet 7 - Implémentez un modèle de scoring
**Prêt à dépenser** est une société financière qui propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt.
L’entreprise souhaite mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données variées. Cet algorithme implémentera un Dashboard interactif à travers d'une API de prédiction pour expliquer de façon la plus transparente possible les décisions d’octroi de crédit.

Liens vers les répertoires Github de ce projet:
* Les notebooks du projet : https://github.com/Rimdata/Implementez_un_modele_de_scoring
* L'API de prédiction : https://github.com/Rimdata/p7_predictapi
* Le tableau de bord : https://github.com/Rimdata/P7_creditapp

# Gestionnaire de crédit - Tableau de bord

Le Dashboard est en ligne sur https://credit-app.herokuapp.com/

Ce dépôt GitHub contient un tableau de bord interactif basé sur Streamlit qui permet de prédire la probabilité de remboursement d'un crédit et de classifier la demande de crédit comme accordée ou refusée.
Le tableau de bord utilise un modèle de machine learning LightGBM classifier pour effectuer les prédictions à partir des données clients.

## Fonctionnalités
* Prédiction de la probabilité de remboursement d'un crédit pour un client spécifique en interrogeant l'API de prédiction.
* Classification du client en tant que client à risque ou client fiable.
* Décision d'accorder ou de refuser la demande de crédit.
* Affichage du score client et de l'état de la demande.
* Profil client avec l'importance des variables et l'explication basée sur les valeurs SHAP.
* Visualisation de la position du client par rapport à l'ensemble des clients.
* Histogrammes des variables sélectionnées pour les clients à risque et les clients fiables.
* Graphiques 2D montrant la relation entre deux variables sélectionnées en fonction du score client.

## Utilisation du tableau de bord
Une fois le tableau de bord ouvert, vous pouvez sélectionner un client à partir de la liste déroulante "Identifiant Client" dans la barre latérale.
Vous pouvez également choisir les variables spécifiques que vous souhaitez afficher en utilisant les options de sélection multiple dans la barre latérale.

Le tableau de bord affichera ensuite les informations sur le client sélectionné, y compris la probabilité de remboursement, l'état de la demande et le score client. 
Vous pourrez également visualiser le profil client, l'importance des variables, la position du client par rapport à l'ensemble des clients et les graphiques des variables sélectionnées.

## Pour le bon fonctionnement de l'API

* creditapp.py : code Python du tableau de board avec Streamlit
* requirements.txt : un fichier listant les packages utilisés 
* test_app.py : code Python avec les tests unitaires 
* Procfile
* df_credit_dash_score.csv : les données des clients à tester
