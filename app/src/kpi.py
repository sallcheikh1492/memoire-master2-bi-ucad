"""Calcul des indicateurs de performance (KPI) opErationnels et de qualitE de service."""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config


def global_kpis(df: pd.DataFrame) -> dict:
    """KPI synthEtiques pour la vue exEcutive."""
    real = df[df["phantom"] == 0]            # on exclut les appels fantOmes
    n = len(real)
    served = real["served"].sum()
    aband = real["abandoned"].sum()
    queued = real[real["entered_queue"] == 1]
    sl = service_level(real)
    return {
        "appels_total": int(n),
        "appels_servis": int(served),
        "appels_abandonnes": int(aband),
        "taux_abandon": round(100 * aband / n, 2) if n else 0.0,
        "taux_service": round(100 * served / n, 2) if n else 0.0,
        "aht_moyen_s": round(real["aht"].mean(), 1),
        "attente_moyenne_s": round(queued["wait_time"].mean(), 1) if len(queued) else 0.0,
        "service_level_pct": round(100 * sl, 1),
        "nb_agents": int(real.loc[real["server"].notna(), "server"].nunique()),
        "appels_fantomes": int(df["phantom"].sum()),
    }


def service_level(df: pd.DataFrame, threshold_s: int = config.DEFAULT_SLA_SECONDS) -> float:
    """Part des appels rEpondus dans le dElai cible (servis ET attente <= seuil)."""
    real = df[df["phantom"] == 0]
    if len(real) == 0:
        return 0.0
    answered_in_time = ((real["served"] == 1) & (real["wait_time"] <= threshold_s)).sum()
    return answered_in_time / len(real)


def kpis_by_type(df: pd.DataFrame) -> pd.DataFrame:
    real = df[df["phantom"] == 0].copy()
    g = real.groupby("type_label").agg(
        appels=("call_id", "count"),
        taux_abandon=("abandoned", lambda s: round(100 * s.mean(), 2)),
        aht_moyen=("aht", "mean"),
        attente_moyenne=("wait_time", "mean"),
    ).reset_index()
    g["aht_moyen"] = g["aht_moyen"].round(1)
    g["attente_moyenne"] = g["attente_moyenne"].round(1)
    return g.sort_values("appels", ascending=False)


def daily_volume(df: pd.DataFrame) -> pd.DataFrame:
    """SErie journaliEre du volume d'appels (arrivEes rEelles, hors fantOmes)."""
    real = df[df["phantom"] == 0]
    s = real.groupby("date_only").size().rename("volume")
    full = s.asfreq("D").fillna(0)
    return full.reset_index().rename(columns={"index": "date_only"})


def intraday_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Profil moyen intra-journalier par crEneau de 30 min."""
    real = df[df["phantom"] == 0]
    by_day_slot = real.groupby(["date_only", "slot_label"]).size().rename("v").reset_index()
    prof = by_day_slot.groupby("slot_label")["v"].mean().round(1).rename("volume_moyen")
    return prof.reset_index()


def heatmap_load(df: pd.DataFrame) -> pd.DataFrame:
    """Matrice jour-de-semaine x heure (volume moyen) pour la heatmap de charge."""
    real = df[df["phantom"] == 0].copy()
    pivot = real.pivot_table(
        index="weekday", columns="hour", values="call_id",
        aggfunc="count", fill_value=0,
    )
    # moyenne par occurrence du jour de semaine
    n_weeks = real["date_only"].dt.isocalendar().week.nunique()
    pivot = (pivot / max(n_weeks, 1)).round(1)
    pivot.index = [config.WEEKDAY_LABELS[i] for i in pivot.index]
    return pivot


def abandonment_by_wait_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Taux d'abandon en fonction de tranches de temps d'attente offert (file)."""
    q = df[(df["phantom"] == 0) & (df["entered_queue"] == 1)].copy()
    bins = [0, 10, 30, 60, 120, 300, 600, np.inf]
    labels = ["0-10s", "10-30s", "30-60s", "1-2min", "2-5min", "5-10min", "10min+"]
    q["bucket"] = pd.cut(q["wait_time"], bins=bins, labels=labels, right=False)
    g = q.groupby("bucket", observed=True).agg(
        appels=("call_id", "count"),
        taux_abandon=("abandoned", lambda s: round(100 * s.mean(), 2)),
    ).reset_index()
    return g
