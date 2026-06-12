# 🛡️ CyberShield AI — Détection de Cyberattaques par IA

## 📋 Résumé du Projet

Système complet de détection d'intrusion basé sur le **Machine Learning**.
Analyse le trafic réseau en temps réel et identifie automatiquement 4 types de cyberattaques.

**Précision atteinte : 99.14%**

---

## 🗂️ Structure du Projet

```
cybersec-project/
│
├── ml/
│   ├── train_model.py          # Script d'entraînement rapide
│   ├── cybershield_complete.py # Code Python complet (toutes classes)
│   └── models/
│       ├── rf_model.pkl        # Modèle Random Forest entraîné
│       ├── iso_forest.pkl      # Modèle Isolation Forest
│       ├── scaler.pkl          # Normalisateur StandardScaler
│       └── meta.json           # Métriques et metadata
│
├── backend/
│   └── app.py                  # API REST Flask (Python)
│
├── frontend/
│   └── dashboard.html          # Dashboard web (HTML/CSS/JS)
│
├── database/
│   ├── schema.sql              # Schéma SQL complet
│   └── cybersec.db             # Base de données SQLite (générée)
│
└── README.md                   # Ce fichier
```

---

## ⚙️ Technologies Utilisées

| Composant       | Technologie               | Rôle                                      |
|-----------------|---------------------------|-------------------------------------------|
| IA principale   | Python + scikit-learn     | Random Forest — classification            |
| IA secondaire   | Python + scikit-learn     | Isolation Forest — anomalies inconnues    |
| API Backend     | Flask + Flask-CORS        | REST API (analyse, alertes, statistiques) |
| Base de données | SQLite (compatible MySQL) | Stockage des alertes                      |
| Interface web   | HTML / CSS / JavaScript   | Dashboard temps réel                      |
| Données         | pandas, numpy             | Prétraitement et manipulation             |

---

## 🚀 Lancer le Projet

### 1. Installer les dépendances
```bash
pip install scikit-learn pandas numpy flask flask-cors
```

### 2. Entraîner le modèle
```bash
cd ml/
python train_model.py
```

### 3. Démarrer l'API
```bash
cd backend/
python app.py
# → API disponible sur http://localhost:5000
```

### 4. Ouvrir le Dashboard
```
Ouvrir frontend/dashboard.html dans un navigateur
```

---

## 🔌 Endpoints de l'API

| Méthode | Endpoint             | Description                        |
|---------|---------------------|------------------------------------|
| GET     | /api/health          | Statut de l'API                    |
| GET     | /api/stats           | Statistiques globales              |
| GET     | /api/alerts          | Liste des alertes (avec filtres)   |
| POST    | /api/analyze         | Analyser une connexion             |
| POST    | /api/simulate        | Simuler une attaque                |
| POST    | /api/block/{id}      | Bloquer une IP                     |
| GET     | /api/model-info      | Informations du modèle             |

### Exemple — Analyser une connexion :
```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "duration": 0.05,
    "protocol_type": 0,
    "bytes_sent": 45,
    "bytes_received": 90,
    "nb_connections": 850,
    "error_rate": 0.72,
    "same_ip_count": 600,
    "port_number": 80
  }'
```

### Réponse :
```json
{
  "prediction": "DoS",
  "confidence": 1.0,
  "severity": "critical",
  "is_anomaly": true,
  "probabilities": {
    "Normal": 0.0,
    "DoS": 1.0,
    "Intrusion": 0.0,
    "Malware": 0.0,
    "Phishing": 0.0
  }
}
```

---

## 🧠 Fonctionnement de l'IA

### Chaîne DIKW (vue dans le cours)

```
DONNÉE         →  INFORMATION      →  SAVOIR           →  DÉCISION
Trafic brut       Normalisation       Pattern détecté     Alerte + Blocage
(IP, ports...)    (preprocessing)     (Random Forest)     automatique
```

### Caractéristiques utilisées (Features)

| Feature         | Description                        | Importance |
|-----------------|------------------------------------|------------|
| nb_connections  | Nombre de connexions simultanées   | 31%        |
| same_ip_count   | Répétitions depuis même IP         | 22%        |
| error_rate      | Taux d'erreur réseau               | 18%        |
| bytes_sent      | Octets envoyés                     | 11%        |
| bytes_received  | Octets reçus                       | 8%         |
| port_number     | Port de destination                | 5%         |
| duration        | Durée de connexion (sec)           | 3%         |
| protocol_type   | TCP / UDP / ICMP                   | 2%         |

---

## 📊 Résultats

| Classe     | Précision | Rappel | F1-Score |
|------------|-----------|--------|----------|
| Normal     | 99%       | 100%   | 99%      |
| DoS        | 100%      | 100%   | 100%     |
| Intrusion  | 99%       | 93%    | 96%      |
| Malware    | 100%      | 100%   | 100%     |
| Phishing   | 100%      | 100%   | 100%     |
| **Macro**  | **99%**   | **99%**| **99%**  |

---

## 👥 Équipe

| Nom            | Rôle                              |
|----------------|-----------------------------------|
| Khadija Aaroud | Chef de projet                    |
| Imane Oujoua   | Responsable IA & Machine Learning |
| Inas           | Data Engineering                  |
| Ikhlas         | Développement & Tests ML          |
| Douae          | Base de données                   |
| Ikram          | Interface utilisateur             |
| Alae           | Tests & Rapport final             |

**Encadrant : M. MANI Ayoub**
**Module : Intelligence Artificielle — 2025/2026**
