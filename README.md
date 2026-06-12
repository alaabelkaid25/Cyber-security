CyberShield AI v2.0 - Système Hybride de Détection d'Intrusions Réseau
Ce dépôt contient le cœur algorithmique et le prototype expérimental du projet CyberShield AI v2.0, un pipeline complet de Machine Learning développé en Python pour la classification automatique du trafic réseau et la détection de menaces.

Présentation du Projet
Le programme s'appuie sur le dataset public de référence NSL-KDD pour entraîner et évaluer deux classifieurs supervisés : un Arbre de Décision simple et une Forêt Aléatoire (Random Forest). L'objectif est de distinguer les connexions normales des connexions malveillantes en analysant le comportement statistique des flux réseau.

Ce module valide la logique mathématique et le traitement des données avant l'intégration finale dans une architecture de production comprenant une API REST Flask, une base de données SQLite et un tableau de bord interactif.

Fonctionnalités du Pipeline
Le script Python exécute un pipeline séquentiel structuré en plusieurs étapes clés :

Chargement et préparation des données : Lecture du dataset NSL-KDD au format CSV sans en-tête.

Encodage catégoriel : Transformation automatique des variables textuelles nominales (types de protocoles, services, flags) en variables binaires via l'encodage one-hot pour éviter la multicolinéarité.

Normalisation : Standardisation des variables numériques à l'aide de StandardScaler afin de garantir une contribution équitable des différentes métriques réseau à l'apprentissage.

Division des données : Découpage strict du jeu de données en 80% pour l'entraînement et 20% pour l'évaluation afin de prévenir les fuites de données.

Entraînement des modèles : Ajustement simultané du Decision Tree et du Random Forest (composé de 100 arbres de décision indépendants).

Évaluation : Calcul de la précision globale et génération d'un rapport de classification complet (précision, rappel, F1-Score) par catégorie d'attaque.

Analyse d'importance : Extraction et visualisation des 10 variables réseau les plus discriminantes pour l'identification des menaces.

Architecture Globale du Système
Pour passer du prototype à un système de niveau industriel, CyberShield AI v2.0 s'organise en 4 couches logicielles :

Couche de Collecte : Capture du trafic réseau en temps réel et intégration des sources de données.

Couche de Prétraitement : Nettoyage, transformation et équilibrage des données.

Couche Moteur IA : Classification par Random Forest pour les menaces connues et détection d'anomalies Zero-Day via Isolation Forest.

Couche Interface et Décision : Tableau de bord interactif affichant les métriques de sécurité et les alertes de détection.

Technologies et Dépendances
Le projet est développé en Python et s'appuie sur l'écosystème scientifique standard suivant :

pandas (version supérieure ou égale à 1.3.0) : Pour la manipulation et le nettoyage des structures de données.

scikit-learn (version supérieure ou égale à 0.24.0) : Pour le prétraitement, la normalisation, l'entraînement des modèles et le calcul des métriques.

matplotlib (version supérieure ou égale à 3.4.0) : Pour la génération des graphiques d'importance des fonctionnalités.

numpy (version supérieure ou égale à 1.21.0) : Pour la manipulation des tableaux et le tri des vecteurs de performance.
