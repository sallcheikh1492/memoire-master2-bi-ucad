"""Page 2 — Prévision du volume d'appels (SARIMA / Prophet / naïf saisonnier / LSTM)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src import ui
from src.ui import get_data, PALETTE
from src import kpi, forecasting

st.set_page_config(page_title="Prévisions", page_icon="📈", layout="wide")
ui.inject_css()
ui.hero("Prévision du volume d'appels",
        "Modèles de séries temporelles pour le dimensionnement anticipé des effectifs",
        icon="📈", theme="navy")

df = get_data()

avail = forecasting.available_models()
dispo = ["Naïf saisonnier"] + [k for k in ("SARIMA", "Prophet", "LSTM") if avail[k]]
st.info("Modèles disponibles : " + ", ".join(dispo))
if not avail["Prophet"]:
    st.caption("ℹ️ Prophet est désactivé sur cet environnement (incompatibilité du backend "
               "Stan/DLL sous Windows). Pour l'activer sur un poste sain : variable "
               "d'environnement `MEMOIRE_ENABLE_PROPHET=1`. La comparaison s'appuie sur "
               "SARIMA et LSTM, plus robustes ici.")

# --- Paramètres ------------------------------------------------------------
c1, c2, c3 = st.columns(3)
types = ["(Tous les services)"] + sorted(df["type_label"].dropna().unique())
sel_type = c1.selectbox("Type de service", types)
test_days = c2.slider("Journées de test (hold-out)", 14, 60, 30, step=7)
horizon = c3.slider("Horizon de prévision future (jours)", 7, 30, 14, step=7)

use_lstm = st.checkbox(
    "Inclure le modèle LSTM (réseau de neurones — entraînement plus lent, ~15-30 s)",
    value=avail["LSTM"], disabled=not avail["LSTM"])

data = df if sel_type == "(Tous les services)" else df[df["type_label"] == sel_type]
series = kpi.daily_volume(data).set_index("date_only")["volume"]


@st.cache_data(show_spinner="Entraînement des modèles de prévision...")
def _run(_series_values, _index, test_days, use_lstm):
    s = pd.Series(_series_values, index=pd.DatetimeIndex(_index))
    return forecasting.run_all_models(s, test_days=test_days, use_lstm=use_lstm)


with st.spinner("Calcul..."):
    res = _run(series.values, series.index, test_days, use_lstm)

# --- Tableau comparatif ----------------------------------------------------
ui.section("📊 Comparaison des modèles (hold-out temporel)")
rows = []
for name, r in res["models"].items():
    if "metrics" in r:
        rows.append({"Modèle": name, **r["metrics"]})
comp = pd.DataFrame(rows).sort_values("MAPE_%")
st.dataframe(comp, use_container_width=True, hide_index=True)
best = comp.iloc[0]["Modèle"] if len(comp) else None
if best:
    st.success(f"✅ Meilleur modèle (MAPE le plus faible) : **{best}**")

failed = [name for name, r in res["models"].items() if "error" in r]
if failed:
    st.warning(
        "Modèle(s) indisponible(s) sur cet environnement : **" + ", ".join(failed) + "**. "
        "Cause fréquente : le backend de compilation (Stan/Prophet) n'a pas pu s'exécuter. "
        "Le code reste fonctionnel sur un environnement correctement configuré."
    )

# --- Graphique réel vs prévu ----------------------------------------------
ui.section("📈 Réel vs prévu sur la période de test")
test_idx = res["_test_index"]
test_val = res["_test_values"]
train = res["_train"]

fig = go.Figure()
fig.add_trace(go.Scatter(x=train.index[-60:], y=train.values[-60:],
                         name="Historique", line=dict(color=PALETTE["muted"])))
fig.add_trace(go.Scatter(x=test_idx, y=test_val, name="Réel (test)",
                         line=dict(color="black", width=3)))
colors = [PALETTE["primary"], PALETTE["accent"], PALETTE["ok"], PALETTE["warn"]]
for i, (name, r) in enumerate(res["models"].items()):
    if "forecast" in r:
        fig.add_trace(go.Scatter(x=test_idx, y=r["forecast"], name=name,
                                 line=dict(color=colors[i % len(colors)], dash="dot")))
        if r.get("upper") is not None:
            fig.add_trace(go.Scatter(
                x=list(test_idx) + list(test_idx[::-1]),
                y=list(r["upper"]) + list(r["lower"][::-1]),
                fill="toself", fillcolor="rgba(31,78,121,0.12)",
                line=dict(color="rgba(0,0,0,0)"), name=f"IC 80% {name}",
                showlegend=False))
fig.update_layout(height=450, xaxis_title="Date", yaxis_title="Appels")
st.plotly_chart(fig, use_container_width=True)

# --- Prévision future ------------------------------------------------------
st.divider()
ui.section(f"🔮 Prévision projetée — {horizon} prochains jours")
fc_df, used = forecasting.future_forecast(series, horizon=horizon)
st.caption(f"Modèle utilisé : **{used}**")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=series.index[-30:], y=series.values[-30:],
                          name="Historique récent", line=dict(color=PALETTE["muted"])))
fig2.add_trace(go.Scatter(x=fc_df["date"], y=fc_df["prevision"], name="Prévision",
                          line=dict(color=PALETTE["accent"], width=3)))
if fc_df["ic_haut"].notna().any():
    fig2.add_trace(go.Scatter(
        x=list(fc_df["date"]) + list(fc_df["date"][::-1]),
        y=list(fc_df["ic_haut"]) + list(fc_df["ic_bas"][::-1]),
        fill="toself", fillcolor="rgba(230,126,34,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="IC 80%"))
fig2.update_layout(height=400, xaxis_title="Date", yaxis_title="Appels prévus")
st.plotly_chart(fig2, use_container_width=True)
st.dataframe(fc_df, use_container_width=True, hide_index=True)

st.caption("Astuce : la prévision quotidienne se combine au profil intra-journalier "
           "(page Vue exécutive) pour répartir le volume par créneau de 30 min.")
