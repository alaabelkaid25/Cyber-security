"""
AI-Powered Cybersecurity Threat Detection
==========================================
Module : Machine Learning Engine
Génère des données synthétiques, entraîne un modèle Random Forest
et sauvegarde le modèle pour l'utilisation dans l'API.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import json
import os

# ─── 1. GÉNÉRATION DE DONNÉES SYNTHÉTIQUES ──────────────────────────────────
np.random.seed(42)

def generate_dataset(n_normal=2000, n_attacks=1500):
    """
    Génère un dataset réseau synthétique avec :
    - Trafic normal
    - 4 types d'attaques : DoS, Intrusion, Malware, Phishing
    """
    records = []

    # Trafic NORMAL
    for _ in range(n_normal):
        records.append({
            "duration":        np.random.exponential(2),
            "protocol_type":   np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1]),  # TCP, UDP, ICMP
            "bytes_sent":      np.random.normal(500, 200),
            "bytes_received":  np.random.normal(1200, 400),
            "nb_connections":  np.random.poisson(5),
            "error_rate":      np.random.beta(1, 20),
            "same_ip_count":   np.random.poisson(2),
            "port_number":     np.random.choice([80, 443, 22, 8080], p=[0.4, 0.4, 0.1, 0.1]),
            "label":           0  # normal
        })

    # Attaque DoS (Déni de Service) - connexions massives rapides
    n_dos = n_attacks // 4
    for _ in range(n_dos):
        records.append({
            "duration":        np.random.exponential(0.1),
            "protocol_type":   np.random.choice([0, 2], p=[0.7, 0.3]),
            "bytes_sent":      np.random.normal(50, 10),
            "bytes_received":  np.random.normal(100, 20),
            "nb_connections":  np.random.poisson(500) + 200,
            "error_rate":      np.random.beta(5, 2),
            "same_ip_count":   np.random.poisson(200) + 100,
            "port_number":     np.random.choice([80, 443, 53]),
            "label":           1  # DoS
        })

    # Attaque INTRUSION - accès non autorisé
    n_int = n_attacks // 4
    for _ in range(n_int):
        records.append({
            "duration":        np.random.exponential(15),
            "protocol_type":   np.random.choice([0, 1], p=[0.8, 0.2]),
            "bytes_sent":      np.random.normal(300, 100),
            "bytes_received":  np.random.normal(800, 300),
            "nb_connections":  np.random.poisson(3),
            "error_rate":      np.random.beta(3, 5),
            "same_ip_count":   np.random.poisson(1),
            "port_number":     np.random.choice([22, 23, 3389, 1433]),
            "label":           2  # Intrusion
        })

    # Attaque MALWARE - communication avec serveur C&C
    n_mal = n_attacks // 4
    for _ in range(n_mal):
        records.append({
            "duration":        np.random.normal(30, 5),
            "protocol_type":   np.random.choice([0, 1], p=[0.5, 0.5]),
            "bytes_sent":      np.random.normal(200, 50),
            "bytes_received":  np.random.normal(200, 50),
            "nb_connections":  np.random.poisson(10),
            "error_rate":      np.random.beta(2, 8),
            "same_ip_count":   np.random.poisson(8),
            "port_number":     np.random.choice([6667, 4444, 1337, 8888]),
            "label":           3  # Malware
        })

    # Attaque PHISHING - faux serveur web
    n_phi = n_attacks - n_dos - n_int - n_mal
    for _ in range(n_phi):
        records.append({
            "duration":        np.random.exponential(5),
            "protocol_type":   np.random.choice([0], p=[1.0]),
            "bytes_sent":      np.random.normal(2000, 500),
            "bytes_received":  np.random.normal(5000, 1000),
            "nb_connections":  np.random.poisson(15),
            "error_rate":      np.random.beta(1, 10),
            "same_ip_count":   np.random.poisson(5),
            "port_number":     np.random.choice([80, 443], p=[0.7, 0.3]),
            "label":           4  # Phishing
        })

    df = pd.DataFrame(records)
    # Nettoyage
    df["bytes_sent"]     = df["bytes_sent"].clip(0)
    df["bytes_received"] = df["bytes_received"].clip(0)
    df["error_rate"]     = df["error_rate"].clip(0, 1)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ─── 2. ENTRAÎNEMENT ─────────────────────────────────────────────────────────
print("=" * 55)
print("  AI Cybersecurity - Entraînement du modèle")
print("=" * 55)

df = generate_dataset()
print(f"\n📊 Dataset généré : {len(df)} enregistrements")
print(f"   Répartition :")
labels_map = {0: "Normal", 1: "DoS", 2: "Intrusion", 3: "Malware", 4: "Phishing"}
for k, v in df["label"].value_counts().sort_index().items():
    print(f"   - {labels_map[k]:12s} : {v}")

features = ["duration", "protocol_type", "bytes_sent", "bytes_received",
            "nb_connections", "error_rate", "same_ip_count", "port_number"]

X = df[features].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\n🔧 Entraînement du modèle Random Forest...")
clf = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train_scaled, y_train)

# Isolation Forest pour anomalies inconnues
print("🔧 Entraînement du modèle Isolation Forest...")
iso = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
iso.fit(X_train_scaled[y_train == 0])  # Entraîné uniquement sur le trafic normal

# ─── 3. ÉVALUATION ───────────────────────────────────────────────────────────
y_pred = clf.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Précision globale (Accuracy) : {acc*100:.2f}%")
print("\nRapport de classification :")
print(classification_report(y_test, y_pred, target_names=list(labels_map.values())))

# Feature importance
importances = dict(zip(features, clf.feature_importances_.tolist()))

# ─── 4. SAUVEGARDE ───────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
with open("models/rf_model.pkl",  "wb") as f: pickle.dump(clf, f)
with open("models/scaler.pkl",    "wb") as f: pickle.dump(scaler, f)
with open("models/iso_forest.pkl","wb") as f: pickle.dump(iso, f)

meta = {
    "accuracy":     round(acc * 100, 2),
    "features":     features,
    "labels":       labels_map,
    "importances":  importances,
    "n_train":      int(len(X_train)),
    "n_test":       int(len(X_test))
}
with open("models/meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\n💾 Modèles sauvegardés dans /models/")
print("   - rf_model.pkl     (Random Forest classifier)")
print("   - scaler.pkl       (StandardScaler)")
print("   - iso_forest.pkl   (Isolation Forest)")
print("   - meta.json        (Métriques et metadata)")
print("\n✅ Entraînement terminé avec succès !")
