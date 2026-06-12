-- ============================================================
--  CyberShield AI — Schéma de Base de Données
--  Base : cybersec.db (SQLite) | Compatible MySQL/PostgreSQL
-- ============================================================

-- Table 1 : Alertes de sécurité
CREATE TABLE IF NOT EXISTS alerts (
    id           INTEGER  PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT     NOT NULL,
    ip_source    TEXT     NOT NULL,
    attack_type  TEXT     NOT NULL CHECK(attack_type IN ('Normal','DoS','Intrusion','Malware','Phishing','Anomalie')),
    severity     TEXT     NOT NULL CHECK(severity IN ('none','medium','high','critical')),
    confidence   REAL     NOT NULL CHECK(confidence BETWEEN 0 AND 1),
    bytes_sent   REAL,
    bytes_recv   REAL,
    nb_conn      INTEGER,
    protocol     TEXT     CHECK(protocol IN ('TCP','UDP','ICMP')),
    port_dest    INTEGER,
    blocked      INTEGER  DEFAULT 0,
    blocked_at   TEXT,
    analyst_note TEXT
);

-- Table 2 : Log du trafic (agrégats horaires)
CREATE TABLE IF NOT EXISTS traffic_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    total_conn  INTEGER NOT NULL,
    normal_cnt  INTEGER DEFAULT 0,
    dos_cnt     INTEGER DEFAULT 0,
    intrusion_cnt INTEGER DEFAULT 0,
    malware_cnt INTEGER DEFAULT 0,
    phishing_cnt INTEGER DEFAULT 0,
    normal_pct  REAL,
    threat_pct  REAL
);

-- Table 3 : Modèles IA enregistrés
CREATE TABLE IF NOT EXISTS models (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    version     TEXT    NOT NULL,
    accuracy    REAL    NOT NULL,
    trained_at  TEXT    NOT NULL,
    is_active   INTEGER DEFAULT 0,
    parameters  TEXT,
    notes       TEXT
);

-- Table 4 : IPs bloquées (liste noire)
CREATE TABLE IF NOT EXISTS blocked_ips (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ip_address  TEXT    NOT NULL UNIQUE,
    reason      TEXT,
    blocked_at  TEXT    NOT NULL,
    unblocked_at TEXT,
    is_active   INTEGER DEFAULT 1
);

-- Index pour les requêtes fréquentes
CREATE INDEX IF NOT EXISTS idx_alerts_timestamp  ON alerts(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_type       ON alerts(attack_type);
CREATE INDEX IF NOT EXISTS idx_alerts_severity   ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_ip         ON alerts(ip_source);
CREATE INDEX IF NOT EXISTS idx_blocked_ip        ON blocked_ips(ip_address);

-- Vues utiles
CREATE VIEW IF NOT EXISTS v_attack_summary AS
    SELECT
        attack_type,
        COUNT(*)                          AS total,
        ROUND(AVG(confidence)*100, 1)    AS avg_confidence_pct,
        SUM(blocked)                     AS total_blocked,
        MAX(timestamp)                   AS last_seen
    FROM alerts
    WHERE attack_type != 'Normal'
    GROUP BY attack_type
    ORDER BY total DESC;

CREATE VIEW IF NOT EXISTS v_daily_stats AS
    SELECT
        DATE(timestamp)  AS day,
        COUNT(*)         AS total_alerts,
        SUM(CASE WHEN severity='critical' THEN 1 ELSE 0 END) AS critical_count,
        SUM(blocked)     AS blocked_count
    FROM alerts
    GROUP BY DATE(timestamp)
    ORDER BY day DESC;

-- Données initiales : modèle actif
INSERT OR IGNORE INTO models(name, version, accuracy, trained_at, is_active, parameters, notes)
VALUES (
    'Random Forest Classifier',
    '1.0',
    99.14,
    datetime('now'),
    1,
    '{"n_estimators":150,"max_depth":12,"min_samples_split":5,"class_weight":"balanced","random_state":42}',
    'Modèle principal — entraîné sur dataset synthétique NSL-KDD style, 3500 enregistrements'
);

INSERT OR IGNORE INTO models(name, version, accuracy, trained_at, is_active, parameters, notes)
VALUES (
    'Isolation Forest',
    '1.0',
    NULL,
    datetime('now'),
    1,
    '{"n_estimators":100,"contamination":0.1,"random_state":42}',
    'Modèle secondaire — détection d anomalies inconnues (zero-day attacks)'
);
