"""ETL : extraction, nettoyage, feature engineering du corpus AnonymousBank (1999).

Le fichier brut est un TSV de ~444 000 appels. Il contient des lignes mal formees
(delimiteurs/retours-chariot integres) ainsi que des champs temporels au format H:MM:SS
et des dates au format YYMMDD. Ce module produit une table propre et enrichie, mise en
cache au format parquet pour des chargements ulterieurs instantanes.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from . import config


def _to_seconds(series: pd.Series) -> pd.Series:
    """Convertit une colonne 'H:MM:SS' en secondes (float). Valeurs invalides -> NaN."""
    td = pd.to_timedelta(series, errors="coerce")
    return td.dt.total_seconds()


def _build_report(df_raw_n: int, df_clean: pd.DataFrame) -> dict:
    return {
        "lignes_brutes_lues": df_raw_n,
        "lignes_valides": len(df_clean),
        "taux_retention": round(100 * len(df_clean) / max(df_raw_n, 1), 2),
        "periode_debut": str(df_clean["arrival_dt"].min().date()),
        "periode_fin": str(df_clean["arrival_dt"].max().date()),
        "taux_abandon_pct": round(100 * (df_clean["outcome"] == "HANG").mean(), 2),
        "nb_agents": int(df_clean.loc[df_clean["server"].notna(), "server"].nunique()),
    }


def clean_raw(raw_path: str = config.DATA_RAW) -> pd.DataFrame:
    """Lit le TSV brut, filtre les lignes invalides, type les colonnes et enrichit."""
    # Lecture defensive : tout en str, on saute les lignes au mauvais nombre de champs.
    df = pd.read_csv(
        raw_path, sep="\t", header=0, names=config.RAW_COLUMNS,
        dtype=str, engine="c", on_bad_lines="skip", encoding="latin-1",
    )
    n_raw = len(df)

    # Nettoyage des espaces parasites
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()

    # --- Filtres semantiques (eliminent les lignes decalees) ---------------
    df = df[df["date"].str.fullmatch(r"\d{6}")]                 # date YYMMDD
    df = df[df["outcome"].isin(config.VALID_OUTCOMES)]          # issue valide
    df = df[df["type"].isin(config.VALID_TYPES)]                # type valide
    df = df[df["priority"].isin({"0", "1", "2"})]               # priorite valide

    # --- Typage ------------------------------------------------------------
    df["priority"] = df["priority"].astype(int)
    df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce").fillna(0)
    df["is_identified"] = df["customer_id"] > 0

    for col in ["vru_time", "q_time", "ser_time"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Date + heure d'arrivee (entree VRU = arrivee de l'appel)
    base_date = pd.to_datetime(df["date"], format="%y%m%d", errors="coerce")
    entry_sec = _to_seconds(df["vru_entry"])
    df["arrival_dt"] = base_date + pd.to_timedelta(entry_sec, unit="s")
    df = df[df["arrival_dt"].notna()]

    # Heure de mise en file (pour l'analyse d'abandon)
    df["q_start_dt"] = base_date + pd.to_timedelta(_to_seconds(df["q_start"]), unit="s")

    # --- Variables cibles --------------------------------------------------
    df["served"] = (df["outcome"] == "AGENT").astype(int)
    df["abandoned"] = (df["outcome"] == "HANG").astype(int)
    df["phantom"] = (df["outcome"] == "PHANTOM").astype(int)
    df["entered_queue"] = (df["q_time"].fillna(0) > 0).astype(int)
    df["wait_time"] = df["q_time"].fillna(0)
    df["aht"] = np.where(df["served"] == 1, df["ser_time"], np.nan)  # AHT = duree de service

    # --- Feature engineering temporel -------------------------------------
    dt = df["arrival_dt"].dt
    df["hour"] = dt.hour
    df["weekday"] = dt.weekday
    df["weekday_name"] = df["weekday"].map(config.WEEKDAY_LABELS)
    df["is_weekend"] = df["weekday"].isin(config.ISRAELI_WEEKEND).astype(int)
    df["month"] = dt.month
    df["week"] = dt.isocalendar().week.astype(int)
    df["date_only"] = df["arrival_dt"].dt.normalize()
    minutes = dt.hour * 60 + dt.minute
    df["slot_min"] = (minutes // config.SLOT_MINUTES) * config.SLOT_MINUTES
    df["slot_label"] = (df["slot_min"] // 60).astype(str).str.zfill(2) + ":" + \
                       (df["slot_min"] % 60).astype(str).str.zfill(2)

    # Charge offerte : nb d'appels arrives dans le meme creneau de 30 min (sans fuite de cible)
    load = df.groupby(["date_only", "slot_min"])["call_id"].transform("count")
    df["offered_load_30min"] = load

    # Libelles
    df["type_label"] = df["type"].map(config.TYPE_LABELS)
    df["outcome_label"] = df["outcome"].map(config.OUTCOME_LABELS)

    # Nettoyage des durees aberrantes (services > 2h = artefact)
    df.loc[df["aht"] > 7200, "aht"] = np.nan

    df = df.reset_index(drop=True)
    df.attrs["report"] = _build_report(n_raw, df)
    return df


def load_clean(force: bool = False) -> pd.DataFrame:
    """Charge la table propre depuis le cache parquet, ou la reconstruit si absente."""
    if (not force) and os.path.exists(config.DATA_CLEAN):
        df = pd.read_parquet(config.DATA_CLEAN)
        df.attrs["report"] = _build_report(len(df), df)
        return df
    df = clean_raw()
    try:
        df.to_parquet(config.DATA_CLEAN, index=False)
    except Exception:
        pass  # le cache est facultatif
    return df


if __name__ == "__main__":
    d = load_clean(force=True)
    print(d.attrs["report"])
    print(d[["arrival_dt", "type", "outcome", "wait_time", "aht", "slot_label"]].head())
