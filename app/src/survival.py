"""Analyse de survie de la PATIENCE client (Brown et al., 2005).

PrincipE : pour un appel mis en file, on observe la "patience" du client.
- Si l'appel est ABANDONNE (HANG) : l'EvEnement est observE, la durEe = temps d'attente.
- Si l'appel est SERVI (AGENT) avant abandon : la patience est *censurEe A droite* A son
  temps d'attente (le client aurait patientE au moins ce temps).

On estime ainsi la fonction de survie de la patience S(t) = P(patience > t) par
Kaplan-Meier, et les facteurs de risque d'abandon par un modEle de Cox (hasards proportionnels).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config

try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test
    _HAS_LIFELINES = True
except Exception:
    _HAS_LIFELINES = False

NUM_COV = ["hour", "weekday", "is_weekend", "offered_load_30min"]
CAT_COV = ["type", "priority"]


def has_lifelines() -> bool:
    return _HAS_LIFELINES


def build_survival_data(df: pd.DataFrame, max_wait: int = 1200) -> pd.DataFrame:
    """Restreint aux appels rEels entrEs en file ; dEfinit (durEe, EvEnement)."""
    d = df[(df["phantom"] == 0) & (df["entered_queue"] == 1)].copy()
    d["duration"] = d["wait_time"].astype(float)
    d = d[(d["duration"] > 0) & (d["duration"] <= max_wait)]
    d["event"] = d["abandoned"].astype(int)   # 1 = abandon observE ; 0 = censurE (servi)
    return d


def km_overall(d: pd.DataFrame):
    """Kaplan-Meier global. Renvoie (courbe DataFrame[t, survie, ic_bas, ic_haut], mEdiane)."""
    kmf = KaplanMeierFitter()
    kmf.fit(d["duration"], d["event"], label="survie")
    sf = kmf.survival_function_.copy()
    ci = kmf.confidence_interval_.copy()
    out = pd.DataFrame({
        "t": sf.index.values,
        "survie": sf["survie"].values,
        "ic_bas": ci.iloc[:, 0].values,
        "ic_haut": ci.iloc[:, 1].values,
    })
    return out, float(kmf.median_survival_time_)


def km_by_group(d: pd.DataFrame, group_col: str, label_map: dict = None, min_n: int = 300):
    """Courbes KM stratifiEes par modalitE d'une variable + mEdianes par groupe."""
    curves, medians, sizes = {}, {}, {}
    for g, sub in d.groupby(group_col):
        if len(sub) < min_n:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(sub["duration"], sub["event"])
        name = label_map.get(g, str(g)) if label_map else str(g)
        sf = kmf.survival_function_
        curves[name] = pd.DataFrame({"t": sf.index.values, "survie": sf.iloc[:, 0].values})
        medians[name] = float(kmf.median_survival_time_)
        sizes[name] = int(len(sub))
    return curves, medians, sizes


def survival_summary(d: pd.DataFrame) -> dict:
    """Statistiques descriptives de patience."""
    aband = d[d["event"] == 1]["duration"]
    return {
        "n_appels_file": int(len(d)),
        "n_abandons": int(d["event"].sum()),
        "taux_abandon_pct": round(100 * d["event"].mean(), 2),
        "patience_mediane_abandon_s": round(float(aband.median()), 1) if len(aband) else np.nan,
        "abandon_moins_10s_pct": round(100 * (aband < 10).mean(), 1) if len(aband) else np.nan,
        "abandon_moins_30s_pct": round(100 * (aband < 30).mean(), 1) if len(aband) else np.nan,
    }


def cox_model(d: pd.DataFrame, sample: int = 60000, seed: int = 42):
    """ModEle de Cox (hasards proportionnels). Renvoie (résumé HR DataFrame, métriques)."""
    cols = ["duration", "event"] + NUM_COV + CAT_COV
    data = d[cols].dropna().copy()
    if len(data) > sample:
        data = data.sample(sample, random_state=seed)

    # Standardisation de la charge (Echelle trEs diffErente) pour la convergence
    data["offered_load_30min"] = (
        (data["offered_load_30min"] - data["offered_load_30min"].mean())
        / data["offered_load_30min"].std())

    data["type"] = data["type"].astype(str)
    data["priority"] = "P" + data["priority"].astype(str)
    design = pd.get_dummies(data, columns=CAT_COV, drop_first=True).astype(float)

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(design, duration_col="duration", event_col="event", show_progress=False)

    s = cph.summary.reset_index()
    res = pd.DataFrame({
        "variable": s["covariate"],
        "HR": s["exp(coef)"].round(3),
        "HR_bas": s["exp(coef) lower 95%"].round(3),
        "HR_haut": s["exp(coef) upper 95%"].round(3),
        "p_value": s["p"].round(4),
    }).sort_values("HR", ascending=False).reset_index(drop=True)

    metrics = {"concordance": round(float(cph.concordance_index_), 4),
               "n_obs": int(len(design)), "n_events": int(design["event"].sum())}
    return res, metrics


def logrank_by_group(d: pd.DataFrame, group_col: str):
    """Test du log-rank entre les deux modalitEs les plus frEquentes d'une variable."""
    top = d[group_col].value_counts().head(2).index.tolist()
    if len(top) < 2:
        return None
    a = d[d[group_col] == top[0]]
    b = d[d[group_col] == top[1]]
    r = logrank_test(a["duration"], b["duration"], a["event"], b["event"])
    return {"groupes": [str(top[0]), str(top[1])], "p_value": round(float(r.p_value), 6),
            "statistique": round(float(r.test_statistic), 2)}
