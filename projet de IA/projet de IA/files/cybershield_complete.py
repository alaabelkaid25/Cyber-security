"""
╔══════════════════════════════════════════════════════════════╗
║         CyberShield AI — Code Python Complet                ║
║         Détection de Cyberattaques par Machine Learning      ║
╚══════════════════════════════════════════════════════════════╝

Structure :
  1. Génération du dataset
  2. Prétraitement (preprocessing)
  3. Entraînement (Random Forest + Isolation Forest)
  4. Évaluation et métriques
  5. Prédiction en temps réel
  6. Interface console pour démonstration
"""

# ─── IMPORTS ─────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
import pickle
import json
import os
import sqlite3
import time
import random
from datetime import datetime

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
CONFIG = {
    "n_estimators":     150,
    "max_depth":        12,
    "min_samples_split": 5,
    "class_weight":     "balanced",
    "random_state":     42,
    "test_size":        0.2,
    "n_samples_normal": 2000,
    "n_samples_attack": 1500,
}

LABELS = {0: "Normal", 1: "DoS", 2: "Intrusion", 3: "Malware", 4: "Phishing"}
FEATURES = [
    "duration", "protocol_type", "bytes_sent", "bytes_received",
    "nb_connections", "error_rate", "same_ip_count", "port_number"
]

# ═══════════════════════════════════════════════════════════════
#  MODULE 1 : GÉNÉRATION DES DONNÉES
# ═══════════════════════════════════════════════════════════════
class DataGenerator:
    """
    Génère un dataset réseau synthétique.
    Chaque ligne = une connexion réseau avec ses caractéristiques.
    """

    def __init__(self, seed=42):
        np.random.seed(seed)

    def generate(self, n_normal=2000, n_attacks=1500):
        """
        Génère n_normal enregistrements normaux + n_attacks attaques.

        Retourne : pandas DataFrame avec colonnes = FEATURES + 'label'
        """
        records = []
        records += self._normal(n_normal)
        n_each   = n_attacks // 4
        records += self._dos(n_each)
        records += self._intrusion(n_each)
        records += self._malware(n_each)
        records += self._phishing(n_attacks - 3*n_each)

        df = pd.DataFrame(records)
        # Nettoyage des valeurs impossibles
        df["bytes_sent"]     = df["bytes_sent"].clip(lower=0)
        df["bytes_received"] = df["bytes_received"].clip(lower=0)
        df["error_rate"]     = df["error_rate"].clip(0, 1)
        return df.sample(frac=1, random_state=42).reset_index(drop=True)

    def _normal(self, n):
        return [{
            "duration":        np.random.exponential(2),
            "protocol_type":   np.random.choice([0,1,2], p=[0.6,0.3,0.1]),
            "bytes_sent":      np.random.normal(500, 200),
            "bytes_received":  np.random.normal(1200, 400),
            "nb_connections":  np.random.poisson(5),
            "error_rate":      np.random.beta(1, 20),
            "same_ip_count":   np.random.poisson(2),
            "port_number":     np.random.choice([80,443,22,8080], p=[0.4,0.4,0.1,0.1]),
            "label": 0
        } for _ in range(n)]

    def _dos(self, n):
        """DoS : connexions massives, courte durée, taux d'erreur élevé"""
        return [{
            "duration":        np.random.exponential(0.1),
            "protocol_type":   np.random.choice([0,2], p=[0.7,0.3]),
            "bytes_sent":      np.random.normal(50, 10),
            "bytes_received":  np.random.normal(100, 20),
            "nb_connections":  np.random.poisson(500) + 200,
            "error_rate":      np.random.beta(5, 2),
            "same_ip_count":   np.random.poisson(200) + 100,
            "port_number":     np.random.choice([80,443,53]),
            "label": 1
        } for _ in range(n)]

    def _intrusion(self, n):
        """Intrusion : durée longue, ports sensibles (SSH, RDP)"""
        return [{
            "duration":        np.random.exponential(15),
            "protocol_type":   np.random.choice([0,1], p=[0.8,0.2]),
            "bytes_sent":      np.random.normal(300, 100),
            "bytes_received":  np.random.normal(800, 300),
            "nb_connections":  np.random.poisson(3),
            "error_rate":      np.random.beta(3, 5),
            "same_ip_count":   np.random.poisson(1),
            "port_number":     np.random.choice([22,23,3389,1433]),
            "label": 2
        } for _ in range(n)]

    def _malware(self, n):
        """Malware : trafic périodique symétrique, ports suspects"""
        return [{
            "duration":        np.random.normal(30, 5),
            "protocol_type":   np.random.choice([0,1], p=[0.5,0.5]),
            "bytes_sent":      np.random.normal(200, 50),
            "bytes_received":  np.random.normal(200, 50),
            "nb_connections":  np.random.poisson(10),
            "error_rate":      np.random.beta(2, 8),
            "same_ip_count":   np.random.poisson(8),
            "port_number":     np.random.choice([6667,4444,1337,8888]),
            "label": 3
        } for _ in range(n)]

    def _phishing(self, n):
        """Phishing : beaucoup de données sur port 80, durée courte"""
        return [{
            "duration":        np.random.exponential(5),
            "protocol_type":   0,  # Toujours TCP
            "bytes_sent":      np.random.normal(2000, 500),
            "bytes_received":  np.random.normal(5000, 1000),
            "nb_connections":  np.random.poisson(15),
            "error_rate":      np.random.beta(1, 10),
            "same_ip_count":   np.random.poisson(5),
            "port_number":     np.random.choice([80,443], p=[0.7,0.3]),
            "label": 4
        } for _ in range(n)]


# ═══════════════════════════════════════════════════════════════
#  MODULE 2 : PRÉTRAITEMENT
# ═══════════════════════════════════════════════════════════════
class Preprocessor:
    """
    Nettoie et transforme les données brutes.
    Passage : Données → Information (chaîne DIKW)
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted  = False

    def fit_transform(self, X):
        """Entraîne le scaler et normalise les données."""
        X_scaled = self.scaler.fit_transform(X)
        self.fitted = True
        return X_scaled

    def transform(self, X):
        """Normalise de nouvelles données (après entraînement)."""
        if not self.fitted:
            raise RuntimeError("Preprocessor pas encore entraîné. Appeler fit_transform() d'abord.")
        return self.scaler.transform(X)

    def get_feature_stats(self, df, features):
        """Retourne les statistiques descriptives du dataset."""
        return df[features].describe()


# ═══════════════════════════════════════════════════════════════
#  MODULE 3 : MODÈLE IA
# ═══════════════════════════════════════════════════════════════
class CyberThreatDetector:
    """
    Système de détection des cyberattaques.
    Combine Random Forest (classification) + Isolation Forest (anomalies).
    """

    def __init__(self, config=CONFIG):
        self.config = config
        self.clf    = None  # Modèle Random Forest
        self.iso    = None  # Modèle Isolation Forest
        self.prep   = Preprocessor()
        self.trained = False

    def train(self, X_train, y_train):
        """
        Entraîne les deux modèles.
        X_train : array de features (brutes, non normalisées)
        y_train : labels (0=Normal, 1=DoS, 2=Intrusion, 3=Malware, 4=Phishing)
        """
        X_scaled = self.prep.fit_transform(X_train)

        # ── Random Forest ──────────────────────────────────
        print("  Entraînement Random Forest...")
        self.clf = RandomForestClassifier(
            n_estimators     = self.config["n_estimators"],
            max_depth        = self.config["max_depth"],
            min_samples_split= self.config["min_samples_split"],
            class_weight     = self.config["class_weight"],
            random_state     = self.config["random_state"],
            n_jobs           = -1
        )
        self.clf.fit(X_scaled, y_train)

        # ── Isolation Forest ───────────────────────────────
        # Entraîné uniquement sur le trafic normal
        print("  Entraînement Isolation Forest (anomalies)...")
        X_normal = X_scaled[y_train == 0]
        self.iso = IsolationForest(
            n_estimators  = 100,
            contamination = 0.1,
            random_state  = self.config["random_state"]
        )
        self.iso.fit(X_normal)

        self.trained = True
        return self

    def predict(self, X):
        """
        Prédit la classe d'une connexion réseau.
        Retourne un dict avec : label, confidence, probabilities, is_anomaly
        """
        if not self.trained:
            raise RuntimeError("Modèle non entraîné.")

        X_scaled = self.prep.transform(X)

        # Prédiction Random Forest
        pred   = self.clf.predict(X_scaled)[0]
        proba  = self.clf.predict_proba(X_scaled)[0]
        label  = LABELS[int(pred)]
        conf   = float(proba[int(pred)])

        # Détection d'anomalie Isolation Forest
        iso_pred = self.iso.predict(X_scaled)[0]  # -1 = anomalie
        iso_score = float(self.iso.score_samples(X_scaled)[0])

        return {
            "label":         label,
            "confidence":    round(conf, 4),
            "probabilities": {LABELS[i]: round(float(p),4) for i,p in enumerate(proba)},
            "is_anomaly":    iso_pred == -1,
            "anomaly_score": round(iso_score, 4),
        }

    def evaluate(self, X_test, y_test):
        """
        Évalue le modèle sur les données de test.
        Retourne les métriques de performance.
        """
        X_scaled = self.prep.transform(X_test)
        y_pred   = self.clf.predict(X_scaled)
        acc      = accuracy_score(y_test, y_pred)

        print("\n" + "="*55)
        print(f"  Précision globale (Accuracy) : {acc*100:.2f}%")
        print("="*55)
        print(classification_report(
            y_test, y_pred,
            target_names=[LABELS[i] for i in sorted(LABELS)]
        ))

        return {
            "accuracy":  round(acc, 4),
            "f1_macro":  round(f1_score(y_test, y_pred, average='macro'), 4),
            "recall":    round(recall_score(y_test, y_pred, average='macro'), 4),
            "precision": round(precision_score(y_test, y_pred, average='macro'), 4),
        }

    def feature_importance(self, features=FEATURES):
        """Retourne l'importance de chaque feature dans le modèle."""
        if not self.trained:
            return {}
        return dict(sorted(
            zip(features, self.clf.feature_importances_),
            key=lambda x: x[1], reverse=True
        ))

    def save(self, path="models"):
        """Sauvegarde les modèles dans le dossier spécifié."""
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/rf_model.pkl",   "wb") as f: pickle.dump(self.clf,        f)
        with open(f"{path}/iso_forest.pkl", "wb") as f: pickle.dump(self.iso,        f)
        with open(f"{path}/scaler.pkl",     "wb") as f: pickle.dump(self.prep.scaler,f)
        print(f"✅ Modèles sauvegardés dans '{path}/'")

    @classmethod
    def load(cls, path="models"):
        """Charge les modèles sauvegardés."""
        detector = cls()
        with open(f"{path}/rf_model.pkl",   "rb") as f: detector.clf         = pickle.load(f)
        with open(f"{path}/iso_forest.pkl", "rb") as f: detector.iso         = pickle.load(f)
        with open(f"{path}/scaler.pkl",     "rb") as f: detector.prep.scaler = pickle.load(f)
        detector.prep.fitted = True
        detector.trained     = True
        print(f"✅ Modèles chargés depuis '{path}/'")
        return detector


# ═══════════════════════════════════════════════════════════════
#  MODULE 4 : BASE DE DONNÉES
# ═══════════════════════════════════════════════════════════════
class AlertDatabase:
    """
    Gestion de la base de données SQLite.
    Stocke toutes les alertes et l'historique des détections.
    """

    def __init__(self, db_path="database/cybersec.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init()

    def _init(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT    NOT NULL,
                ip_source   TEXT    NOT NULL,
                attack_type TEXT    NOT NULL,
                severity    TEXT    NOT NULL,
                confidence  REAL    NOT NULL,
                bytes_sent  REAL,
                nb_conn     INTEGER,
                protocol    TEXT,
                blocked     INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()

    def insert(self, ip, attack_type, severity, confidence, bytes_sent, nb_conn, protocol):
        """Insère une nouvelle alerte dans la base de données."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO alerts (timestamp,ip_source,attack_type,severity,confidence,bytes_sent,nb_conn,protocol)
            VALUES (?,?,?,?,?,?,?,?)
        """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              ip, attack_type, severity, round(confidence,4),
              bytes_sent, nb_conn, protocol))
        conn.commit()
        conn.close()

    def get_stats(self):
        """Retourne les statistiques globales."""
        conn = sqlite3.connect(self.db_path)
        total    = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        by_type  = conn.execute("SELECT attack_type, COUNT(*) FROM alerts GROUP BY attack_type").fetchall()
        critical = conn.execute("SELECT COUNT(*) FROM alerts WHERE severity='critical'").fetchone()[0]
        conn.close()
        return {"total": total, "critical": critical, "by_type": dict(by_type)}

    def get_recent(self, n=10):
        """Retourne les n dernières alertes."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?", (n,)
        ).fetchall()
        conn.close()
        return rows


# ═══════════════════════════════════════════════════════════════
#  MODULE 5 : DÉMONSTRATION EN CONSOLE
# ═══════════════════════════════════════════════════════════════
SEVERITY_MAP = {"Normal":"none","DoS":"critical","Intrusion":"high","Malware":"high","Phishing":"medium"}
COLORS = {"critical":"\033[91m","high":"\033[93m","medium":"\033[33m","none":"\033[92m","reset":"\033[0m","cyan":"\033[96m","bold":"\033[1m"}

def colored(text, color):
    return f"{COLORS.get(color,'')}{text}{COLORS['reset']}"

def demo_scenario(detector):
    """
    Démonstration interactive en console.
    Simule plusieurs types de connexions et affiche les résultats.
    """
    scenarios = [
        ("Navigation normale",   [2.5,  0, 480,  1100,  4,   0.02, 2,  443]),
        ("Attaque DoS",          [0.05, 0,  45,    90, 850,  0.72, 600, 80]),
        ("Intrusion SSH",        [22.0, 0, 310,   750,   2,  0.35,   1,  22]),
        ("Communication Malware",[31.5, 1, 195,   200,  12,  0.15,   9,4444]),
        ("Serveur Phishing",     [4.8,  0,1950,  4900,  18,  0.08,   6,  80]),
    ]

    print("\n" + colored("═"*60, "cyan"))
    print(colored("  DÉMONSTRATION — ANALYSE EN TEMPS RÉEL", "bold"))
    print(colored("═"*60, "cyan"))

    for name, feat in scenarios:
        print(f"\n  Connexion : {colored(name, 'bold')}")
        print(f"  Features  : nb_conn={feat[4]}, error={feat[5]:.2f}, port={int(feat[6+1])}")

        result = detector.predict(np.array([feat]))
        label = result["label"]
        conf  = result["confidence"]
        sev   = SEVERITY_MAP[label]
        anom  = "⚠ ANOMALIE" if result["is_anomaly"] else "OK"

        if sev == "none":
            verdict = colored(f"✓ {label}", "none")
        elif sev == "critical":
            verdict = colored(f"🚨 {label} — CRITIQUE", "critical")
        elif sev == "high":
            verdict = colored(f"⚠  {label} — ÉLEVÉ", "high")
        else:
            verdict = colored(f"!  {label} — MOYEN", "medium")

        print(f"  Résultat  : {verdict}")
        print(f"  Confiance : {conf*100:.1f}% | Isolation Forest : {anom}")
        time.sleep(0.3)

    print("\n" + colored("═"*60, "cyan"))


# ═══════════════════════════════════════════════════════════════
#  MAIN — PIPELINE COMPLET
# ═══════════════════════════════════════════════════════════════
def main():
    print(colored("\n╔══════════════════════════════════════════════════════╗", "cyan"))
    print(colored("║     CyberShield AI — Pipeline Complet               ║", "cyan"))
    print(colored("╚══════════════════════════════════════════════════════╝", "cyan"))

    # 1. Génération des données
    print(colored("\n[1] Génération du dataset...", "bold"))
    gen = DataGenerator(seed=CONFIG["random_state"])
    df  = gen.generate(
        n_normal  = CONFIG["n_samples_normal"],
        n_attacks = CONFIG["n_samples_attack"]
    )
    print(f"    {len(df)} enregistrements générés.")
    for lbl, cnt in df["label"].value_counts().sort_index().items():
        print(f"    - {LABELS[lbl]:12s} : {cnt}")

    # 2. Séparation features / labels
    X = df[FEATURES].values
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = CONFIG["test_size"],
        random_state = CONFIG["random_state"],
        stratify     = y
    )
    print(f"\n    Train : {len(X_train)} | Test : {len(X_test)}")

    # 3. Entraînement
    print(colored("\n[2] Entraînement des modèles...", "bold"))
    detector = CyberThreatDetector(CONFIG)
    detector.train(X_train, y_train)

    # 4. Évaluation
    print(colored("\n[3] Évaluation des performances...", "bold"))
    metrics = detector.evaluate(X_test, y_test)

    print("\n  Importance des features :")
    for feat, imp in detector.feature_importance().items():
        bar = "█" * int(imp * 30)
        print(f"    {feat:20s} {bar} {imp*100:.1f}%")

    # 5. Sauvegarde
    print(colored("\n[4] Sauvegarde des modèles...", "bold"))
    detector.save("models")

    # 6. Base de données
    print(colored("\n[5] Initialisation de la base de données...", "bold"))
    db = AlertDatabase("database/cybersec.db")
    print("    Base de données initialisée.")

    # 7. Démonstration
    print(colored("\n[6] Démonstration des prédictions...", "bold"))
    demo_scenario(detector)

    # 8. Résumé
    print(colored("\n╔══════════════════════════════════════════════════════╗", "cyan"))
    print(colored("║  RÉSUMÉ DU PROJET                                   ║", "cyan"))
    print(colored("╠══════════════════════════════════════════════════════╣", "cyan"))
    print(f"║  Accuracy (Random Forest)  : {metrics['accuracy']*100:.2f}%                 ║")
    print(f"║  F1-Score (macro avg)      : {metrics['f1_macro']*100:.2f}%                 ║")
    print(f"║  Recall   (macro avg)      : {metrics['recall']*100:.2f}%                 ║")
    print(f"║  Classes détectées         : Normal, DoS, Intrusion,    ║")
    print(f"║                              Malware, Phishing           ║")
    print(f"║  Anomalies inconnues       : Isolation Forest           ║")
    print(colored("╚══════════════════════════════════════════════════════╝", "cyan"))
    print()


if __name__ == "__main__":
    main()
