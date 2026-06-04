"""Analyse de la performance des agents : KPI individuels + segmentation (clustering)."""
from __future__ import annotations
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def agent_kpis(df: pd.DataFrame, min_calls: int = 100) -> pd.DataFrame:
    """KPI par agent (sur les appels effectivement servis)."""
    served = df[(df["served"] == 1) & (df["server"].notna()) & (df["server"] != "NO_SERVER")].copy()
    g = served.groupby("server").agg(
        appels_traites=("call_id", "count"),
        aht_moyen=("aht", "mean"),
        aht_median=("aht", "median"),
        jours_actifs=("date_only", "nunique"),
    ).reset_index()
    # ProductivitE : appels traitEs par jour actif
    g["productivite_jour"] = (g["appels_traites"] / g["jours_actifs"]).round(1)
    g["aht_moyen"] = g["aht_moyen"].round(1)
    g["aht_median"] = g["aht_median"].round(1)
    # Part servie dans le dElai (proxy temps de rEponse via attente)
    wait = df[(df["served"] == 1) & (df["server"].notna())]
    sl = wait.groupby("server")["wait_time"].mean().round(1).rename("attente_moyenne_s")
    g = g.merge(sl, on="server", how="left")
    g = g[g["appels_traites"] >= min_calls].reset_index(drop=True)
    return g.sort_values("appels_traites", ascending=False)


def segment_agents(agents_df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Segmente les agents en profils de performance par K-means."""
    if len(agents_df) < k:
        agents_df = agents_df.copy()
        agents_df["segment"] = "Unique"
        return agents_df
    feats = ["aht_moyen", "productivite_jour", "appels_traites", "attente_moyenne_s"]
    X = agents_df[feats].fillna(agents_df[feats].median())
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(Xs)
    out = agents_df.copy()
    out["cluster"] = labels
    # Nommage interpretable des clusters selon productivitE moyenne
    order = out.groupby("cluster")["productivite_jour"].mean().sort_values(ascending=False)
    names = {}
    tiers = ["Top performers", "Performance intermédiaire", "À accompagner"]
    for rank, cl in enumerate(order.index):
        names[cl] = tiers[rank] if rank < len(tiers) else f"Groupe {rank+1}"
    out["segment"] = out["cluster"].map(names)
    return out.drop(columns=["cluster"])


def perf_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """AHT moyen par type de service (facteur explicatif de la performance)."""
    served = df[df["served"] == 1]
    g = served.groupby("type_label").agg(
        appels=("call_id", "count"),
        aht_moyen=("aht", "mean"),
    ).reset_index()
    g["aht_moyen"] = g["aht_moyen"].round(1)
    return g.sort_values("aht_moyen", ascending=False)
