"""Configuration centrale de l'application (chemins, constantes, libellés metier)."""
from __future__ import annotations
import os

# --- Chemins ---------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../app
PROJECT_ROOT = os.path.dirname(APP_DIR)                                  # .../memoire
DATA_RAW = os.path.join(PROJECT_ROOT, "Annee1999.csv")
CACHE_DIR = os.path.join(APP_DIR, "_cache")
DATA_CLEAN = os.path.join(CACHE_DIR, "calls_clean.parquet")

os.makedirs(CACHE_DIR, exist_ok=True)

# --- Schema brut (TSV) -----------------------------------------------------
RAW_COLUMNS = [
    "vru_line", "call_id", "customer_id", "priority", "type", "date",
    "vru_entry", "vru_exit", "vru_time", "q_start", "q_exit", "q_time",
    "outcome", "ser_start", "ser_exit", "ser_time", "server",
]

VALID_OUTCOMES = {"AGENT", "HANG", "PHANTOM"}
VALID_TYPES = {"PS", "PE", "NW", "NE", "IN", "TT", "AA"}

# --- Libelles metier -------------------------------------------------------
TYPE_LABELS = {
    "PS": "Service courant (hébreu)",
    "PE": "Service anglophone",
    "NW": "Prospect / nouveau client",
    "NE": "Service NE",
    "IN": "Support Internet",
    "TT": "Bourse / courtage",
    "AA": "Autre",
}

OUTCOME_LABELS = {
    "AGENT": "Servi par un agent",
    "HANG": "Abandon (raccroché)",
    "PHANTOM": "Appel fantôme",
}

PRIORITY_LABELS = {0: "Standard (0)", 1: "Prioritaire (1)", 2: "Haute priorité (2)"}

# Jours de la semaine (lundi=0). En Israel le week-end est vendredi-samedi.
WEEKDAY_LABELS = {
    0: "Lundi", 1: "Mardi", 2: "Mercredi", 3: "Jeudi",
    4: "Vendredi", 5: "Samedi", 6: "Dimanche",
}
ISRAELI_WEEKEND = {4, 5}  # vendredi, samedi

# --- Parametres par defaut -------------------------------------------------
DEFAULT_SLA_SECONDS = 20      # objectif: repondre en moins de 20 s
DEFAULT_SLA_TARGET = 0.80     # 80 % des appels
SLOT_MINUTES = 30             # granularite infra-journaliere
