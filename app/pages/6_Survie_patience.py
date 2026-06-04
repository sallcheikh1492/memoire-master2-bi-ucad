"""Page 6 — Analyse de survie de la patience client (Kaplan-Meier + Cox)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src import ui
from src.ui import get_data, fmt_int, fmt_sec, PALETTE
from src import survival, config

st.set_page_config(page_title="Survie / patience", page_icon="⏳", layout="wide")
ui.inject_css()
ui.hero("Survie de la patience client",
        "Analyse de survie — modéliser <i>quand</i> un client abandonne, pas seulement <i>s'il</i> abandonne",
        icon="⏳", theme="teal")

df = get_data()

if not survival.has_lifelines():
    st.warning("Librairie `lifelines` non installée. Exécuter : `pip install lifelines`.")
    st.stop()

st.info(
    "**Méthode** (Brown et al., 2005) : la durée étudiée est le temps d'attente en file ; "
    "l'événement est l'abandon. Les appels SERVIS sont *censurés à droite* (le client aurait "
    "patienté au moins jusqu'à sa prise en charge). On estime la fonction de survie de la "
    "patience S(t) = P(patienter au-delà de t)."
)


@st.cache_data(show_spinner="Préparation des données de survie...")
def _surv_data():
    return survival.build_survival_data(df)


@st.cache_data(show_spinner="Estimation Kaplan-Meier...")
def _km_overall(n):
    d = _surv_data()
    sample = d.sample(min(n, len(d)), random_state=1)
    return survival.km_overall(sample)


@st.cache_data(show_spinner="Estimation par groupe...")
def _km_group(col):
    lm = config.TYPE_LABELS if col == "type" else config.PRIORITY_LABELS
    return survival.km_by_group(_surv_data(), col, lm)


@st.cache_data(show_spinner="Ajustement du modèle de Cox...")
def _cox():
    return survival.cox_model(_surv_data())


d = _surv_data()
summ = survival.survival_summary(d)

ui.kpi_row([
    {"icon": "📥", "value": fmt_int(summ['n_appels_file']), "label": "Appels en file", "color": "#16a085"},
    {"icon": "⏱️", "value": fmt_sec(summ["patience_mediane_abandon_s"]), "label": "Patience médiane (abandons)", "color": "#1f4e79"},
    {"icon": "⚡", "value": f"{summ['abandon_moins_10s_pct']} %", "label": "Abandons < 10 s", "color": "#e74c3c"},
    {"icon": "🕐", "value": f"{summ['abandon_moins_30s_pct']} %", "label": "Abandons < 30 s", "color": "#e67e22"},
])

st.write("")

# --- Courbe KM globale -----------------------------------------------------
ui.section("📉 Fonction de survie de la patience (globale)")
sf, med = _km_overall(80000)
sf = sf[sf["t"] <= 600]
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=list(sf["t"]) + list(sf["t"][::-1]),
    y=list(sf["ic_haut"]) + list(sf["ic_bas"][::-1]),
    fill="toself", fillcolor="rgba(22,160,133,0.15)",
    line=dict(color="rgba(0,0,0,0)"), name="IC 95%", showlegend=True))
fig.add_trace(go.Scatter(x=sf["t"], y=sf["survie"], name="S(t)",
                         line=dict(color=PALETTE["primary"], width=3)))
fig.add_hline(y=0.5, line_dash="dot", line_color=PALETTE["muted"], annotation_text="50 %")
fig.update_layout(height=420, xaxis_title="Temps d'attente t (secondes)",
                  yaxis_title="Probabilité de patienter au-delà de t", yaxis_range=[0, 1])
st.plotly_chart(fig, use_container_width=True)
st.caption(f"Patience médiane estimée (incluant la censure) : **{fmt_sec(med)}** — "
           "valeur supérieure à la médiane des seuls abandons, car la majorité des clients "
           "sont servis avant d'abandonner.")

st.divider()

# --- Stratification --------------------------------------------------------
ui.section("👥 Patience selon le segment")
strat = st.radio("Stratifier par", ["Type de service", "Priorité"], horizontal=True)
col = "type" if strat.startswith("Type") else "priority"
curves, medians, sizes = _km_group(col)

g1, g2 = st.columns([3, 2])
with g1:
    figs = go.Figure()
    palette = px.colors.qualitative.Bold
    for i, (name, cv) in enumerate(curves.items()):
        cv = cv[cv["t"] <= 600]
        figs.add_trace(go.Scatter(x=cv["t"], y=cv["survie"], name=name,
                                  line=dict(color=palette[i % len(palette)], width=2.5)))
    figs.add_hline(y=0.5, line_dash="dot", line_color=PALETTE["muted"])
    figs.update_layout(height=420, xaxis_title="Temps d'attente (s)",
                       yaxis_title="S(t)", yaxis_range=[0, 1],
                       legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(figs, use_container_width=True)

with g2:
    med_df = pd.DataFrame({
        "Segment": list(medians.keys()),
        "Patience médiane (s)": [round(v, 0) for v in medians.values()],
        "Effectif": [sizes[k] for k in medians],
    }).sort_values("Patience médiane (s)")
    figm = px.bar(med_df, x="Patience médiane (s)", y="Segment", orientation="h",
                  text="Patience médiane (s)")
    figm.update_traces(marker_color=PALETTE["accent"])
    figm.update_layout(height=340, yaxis={"categoryorder": "total descending"})
    st.plotly_chart(figm, use_container_width=True)
    lr = survival.logrank_by_group(d, col)
    if lr:
        sig = "significative" if lr["p_value"] < 0.05 else "non significative"
        st.caption(f"Test du log-rank ({lr['groupes'][0]} vs {lr['groupes'][1]}) : "
                   f"p = {lr['p_value']} → différence **{sig}**.")

st.caption("Une survie qui chute vite = clients impatients (abandon précoce). "
           "Les segments les moins patients doivent être routés en priorité.")

st.divider()

# --- Modèle de Cox ---------------------------------------------------------
ui.section("🎯 Facteurs de risque d'abandon — modèle de Cox (hasards proportionnels)")
st.caption("Un *hazard ratio* (HR) > 1 augmente le risque instantané d'abandon ; "
           "< 1 le réduit. Variables : charge offerte (standardisée), heure, jour, type, priorité.")
res, m = _cox()

cc1, cc2 = st.columns([3, 2])
with cc1:
    plot_df = res[res["variable"] != "Intercept"].copy()
    figc = go.Figure()
    figc.add_trace(go.Scatter(
        x=plot_df["HR"], y=plot_df["variable"], mode="markers",
        marker=dict(size=9, color=PALETTE["primary"]),
        error_x=dict(type="data", symmetric=False,
                     array=plot_df["HR_haut"] - plot_df["HR"],
                     arrayminus=plot_df["HR"] - plot_df["HR_bas"]),
        name="HR (IC 95%)"))
    figc.add_vline(x=1.0, line_dash="dash", line_color=PALETTE["warn"],
                   annotation_text="HR = 1 (sans effet)")
    figc.update_layout(height=440, xaxis_title="Hazard Ratio (échelle log)",
                       xaxis_type="log", yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(figc, use_container_width=True)

with cc2:
    st.metric("Indice de concordance (C-index)", m["concordance"],
              help="Qualité discriminante du modèle (0,5 = hasard, 1 = parfait)")
    st.metric("Observations", fmt_int(m['n_obs']))
    st.metric("Événements (abandons)", fmt_int(m['n_events']))
    st.dataframe(res, use_container_width=True, hide_index=True, height=240)

st.success(
    "**Apport scientifique (suggestion innovante n°1)** : l'analyse de survie répond à la "
    "question *quand* le client abandonne et quantifie la patience par segment. Elle alimente "
    "directement le modèle Erlang-A (impatience) et la priorisation du routage : les segments "
    "à faible patience (ex. prospects) doivent être décrochés en priorité. "
    "Référence : Brown et al. (2005), *Statistical Analysis of a Telephone Call Center*, JASA."
)
