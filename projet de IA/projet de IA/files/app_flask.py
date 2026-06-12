"""
AI-Powered Cybersecurity Threat Detection
==========================================
Module : API REST Flask + Base de données SQLite
Endpoints : /api/analyze  /api/alerts  /api/stats  /api/simulate
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, json, os, random, time
from datetime import datetime, timedelta
import numpy as np
import sqlite3

app = Flask(__name__)
CORS(app)

BASE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, "../ml/models")
DB_PATH   = os.path.join(BASE, "../database/cybersec.db")

# ─── CHARGEMENT DES MODÈLES ──────────────────────────────────────────────────
with open(f"{MODEL_DIR}/rf_model.pkl",   "rb") as f: clf     = pickle.load(f)
with open(f"{MODEL_DIR}/scaler.pkl",     "rb") as f: scaler  = pickle.load(f)
with open(f"{MODEL_DIR}/iso_forest.pkl", "rb") as f: iso     = pickle.load(f)
with open(f"{MODEL_DIR}/meta.json")            as f: meta    = json.load(f)

LABELS = {0:"Normal", 1:"DoS", 2:"Intrusion", 3:"Malware", 4:"Phishing"}
SEVERITY = {"Normal":"none", "DoS":"critical", "Intrusion":"high", "Malware":"high", "Phishing":"medium"}
COLORS   = {"Normal":"#22c55e","DoS":"#ef4444","Intrusion":"#f97316","Malware":"#a855f7","Phishing":"#eab308"}
IPS = [f"192.168.{random.randint(1,5)}.{random.randint(1,254)}" for _ in range(30)] + \
      [f"10.0.{random.randint(0,3)}.{random.randint(1,254)}"    for _ in range(20)] + \
      ["203.0.113.45","198.51.100.12","185.220.101.56","45.33.32.156","91.108.4.0"]

# ─── BASE DE DONNÉES ─────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_db()
    c = conn.cursor()
    c.execute("""
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
    c.execute("""
        CREATE TABLE IF NOT EXISTS traffic_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            total_conn  INTEGER,
            normal_pct  REAL,
            threat_pct  REAL
        )
    """)
    conn.commit()

    # Pré-remplir avec des données historiques (7 derniers jours)
    c.execute("SELECT COUNT(*) FROM alerts")
    if c.fetchone()[0] == 0:
        _seed_history(c)
        conn.commit()
    conn.close()

def _seed_history(c):
    """Génère 7 jours d'historique pour l'affichage initial."""
    attack_types = ["DoS","Intrusion","Malware","Phishing"]
    now = datetime.now()
    for day_offset in range(7, 0, -1):
        base_dt = now - timedelta(days=day_offset)
        n_events = random.randint(8, 25)
        for _ in range(n_events):
            dt = base_dt + timedelta(
                hours=random.randint(0,23),
                minutes=random.randint(0,59),
                seconds=random.randint(0,59)
            )
            atype = random.choices(attack_types, weights=[40,30,20,10])[0]
            c.execute("""INSERT INTO alerts
                (timestamp, ip_source, attack_type, severity, confidence, bytes_sent, nb_conn, protocol, blocked)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (dt.strftime("%Y-%m-%d %H:%M:%S"),
                 random.choice(IPS), atype, SEVERITY[atype],
                 round(random.uniform(0.75, 0.99), 3),
                 round(random.uniform(50, 5000), 1),
                 random.randint(10, 800),
                 random.choice(["TCP","UDP","ICMP"]),
                 random.randint(0,1))
            )
        # log trafic
        c.execute("""INSERT INTO traffic_log (timestamp, total_conn, normal_pct, threat_pct)
                     VALUES (?,?,?,?)""",
                  (base_dt.strftime("%Y-%m-%d"),
                   random.randint(800,2000),
                   round(random.uniform(70,90), 1),
                   round(random.uniform(10,30), 1)))

# ─── ENDPOINTS ───────────────────────────────────────────────────────────────
@app.route("/api/health")
def health():
    return jsonify({"status":"ok","model_accuracy": meta["accuracy"]})

@app.route("/api/model-info")
def model_info():
    return jsonify(meta)

@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Analyse une connexion réseau et retourne la prédiction."""
    data = request.json or {}
    feat = [
        float(data.get("duration",        random.expovariate(0.5))),
        int(  data.get("protocol_type",   random.choice([0,1,2]))),
        float(data.get("bytes_sent",      abs(random.gauss(500,200)))),
        float(data.get("bytes_received",  abs(random.gauss(1200,400)))),
        int(  data.get("nb_connections",  random.randint(1,20))),
        float(data.get("error_rate",      random.uniform(0,0.2))),
        int(  data.get("same_ip_count",   random.randint(1,10))),
        int(  data.get("port_number",     random.choice([80,443,22,8080]))),
    ]
    X = scaler.transform([feat])
    pred   = int(clf.predict(X)[0])
    proba  = clf.predict_proba(X)[0]
    label  = LABELS[pred]
    conf   = float(proba[pred])
    anomaly = int(iso.predict(X)[0])  # -1 = anomalie inconnue

    result = {
        "prediction":   label,
        "confidence":   round(conf, 4),
        "severity":     SEVERITY[label],
        "color":        COLORS[label],
        "probabilities":{LABELS[i]: round(float(p),4) for i,p in enumerate(proba)},
        "is_anomaly":   anomaly == -1,
        "features_used": dict(zip(meta["features"], feat))
    }

    # Enregistrer les alertes (pas les normaux)
    if label != "Normal" or anomaly == -1:
        conn = get_db()
        conn.execute("""INSERT INTO alerts
            (timestamp,ip_source,attack_type,severity,confidence,bytes_sent,nb_conn,protocol,blocked)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             data.get("ip_source", random.choice(IPS)),
             label if label != "Normal" else "Anomalie",
             "critical" if anomaly == -1 and label == "Normal" else SEVERITY[label],
             round(conf,3), feat[2], feat[4],
             ["TCP","UDP","ICMP"][feat[1]], 0))
        conn.commit(); conn.close()

    return jsonify(result)

@app.route("/api/simulate", methods=["POST"])
def simulate():
    """Simule une attaque aléatoire pour la démonstration."""
    attack = request.json.get("type", "random")
    templates = {
        "normal":    {"duration":2.5,"protocol_type":0,"bytes_sent":480,"bytes_received":1100,"nb_connections":4,"error_rate":0.02,"same_ip_count":2,"port_number":443},
        "dos":       {"duration":0.05,"protocol_type":0,"bytes_sent":45,"bytes_received":90,"nb_connections":850,"error_rate":0.72,"same_ip_count":600,"port_number":80},
        "intrusion": {"duration":22.0,"protocol_type":0,"bytes_sent":310,"bytes_received":750,"nb_connections":2,"error_rate":0.35,"same_ip_count":1,"port_number":22},
        "malware":   {"duration":31.5,"protocol_type":1,"bytes_sent":195,"bytes_received":200,"nb_connections":12,"error_rate":0.15,"same_ip_count":9,"port_number":4444},
        "phishing":  {"duration":4.8,"protocol_type":0,"bytes_sent":1950,"bytes_received":4900,"nb_connections":18,"error_rate":0.08,"same_ip_count":6,"port_number":80},
    }
    if attack == "random":
        attack = random.choice(["dos","intrusion","malware","phishing"])
    body = templates.get(attack, templates["dos"])
    body["ip_source"] = random.choice(IPS)

    # Appel interne
    feat = [body["duration"],body["protocol_type"],body["bytes_sent"],body["bytes_received"],
            body["nb_connections"],body["error_rate"],body["same_ip_count"],body["port_number"]]
    X = scaler.transform([feat])
    pred  = int(clf.predict(X)[0])
    proba = clf.predict_proba(X)[0]
    label = LABELS[pred]
    conf  = float(proba[pred])
    anomaly = int(iso.predict(X)[0])

    conn = get_db()
    conn.execute("""INSERT INTO alerts
        (timestamp,ip_source,attack_type,severity,confidence,bytes_sent,nb_conn,protocol,blocked)
        VALUES (?,?,?,?,?,?,?,?,?)""",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
         body["ip_source"], label if label!="Normal" else "Anomalie",
         SEVERITY[label], round(conf,3), feat[2], feat[4],
         ["TCP","UDP","ICMP"][feat[1]], 0))
    conn.commit(); conn.close()

    return jsonify({
        "simulated_type": attack,
        "prediction": label,
        "confidence": round(conf,4),
        "severity": SEVERITY[label],
        "color": COLORS[label],
        "probabilities": {LABELS[i]: round(float(p),4) for i,p in enumerate(proba)},
        "is_anomaly": anomaly == -1,
        "ip_source": body["ip_source"]
    })

@app.route("/api/alerts")
def get_alerts():
    limit  = int(request.args.get("limit", 50))
    offset = int(request.args.get("offset", 0))
    atype  = request.args.get("type", None)
    conn = get_db()
    if atype and atype != "all":
        rows = conn.execute(
            "SELECT * FROM alerts WHERE attack_type=? ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (atype, limit, offset)).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM alerts WHERE attack_type=?", (atype,)).fetchone()[0]
    else:
        rows = conn.execute(
            "SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (limit, offset)).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    conn.close()
    return jsonify({"alerts": [dict(r) for r in rows], "total": total})

@app.route("/api/stats")
def stats():
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    by_type = conn.execute(
        "SELECT attack_type, COUNT(*) as cnt FROM alerts GROUP BY attack_type ORDER BY cnt DESC"
    ).fetchall()
    recent_24h = conn.execute(
        "SELECT COUNT(*) FROM alerts WHERE timestamp >= datetime('now','-1 day')"
    ).fetchone()[0]
    critical = conn.execute(
        "SELECT COUNT(*) FROM alerts WHERE severity='critical'"
    ).fetchone()[0]
    avg_conf = conn.execute(
        "SELECT AVG(confidence) FROM alerts WHERE attack_type != 'Normal'"
    ).fetchone()[0] or 0
    # Par jour (7 derniers jours)
    daily = conn.execute("""
        SELECT date(timestamp) as day, COUNT(*) as cnt
        FROM alerts WHERE timestamp >= date('now','-7 days')
        GROUP BY day ORDER BY day
    """).fetchall()
    conn.close()
    return jsonify({
        "total_alerts":  total,
        "last_24h":      recent_24h,
        "critical":      critical,
        "avg_confidence": round(avg_conf * 100, 1),
        "by_type":       [dict(r) for r in by_type],
        "daily_trend":   [dict(r) for r in daily],
        "model_accuracy": meta["accuracy"]
    })

@app.route("/api/block/<int:alert_id>", methods=["POST"])
def block_ip(alert_id):
    conn = get_db()
    conn.execute("UPDATE alerts SET blocked=1 WHERE id=?", (alert_id,))
    conn.commit()
    row = conn.execute("SELECT ip_source FROM alerts WHERE id=?", (alert_id,)).fetchone()
    conn.close()
    return jsonify({"success": True, "blocked_ip": row["ip_source"] if row else None})

if __name__ == "__main__":
    init_db()
    print("🚀 API démarrée sur http://localhost:5000")
    app.run(debug=False, port=5000)
