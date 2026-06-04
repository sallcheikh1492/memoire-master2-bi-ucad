"""Page 3 — Performance des agents : KPI individuels et segmentation."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.express as px

from src import ui
from src.ui import get_data, fmt_int, PALETTE
from src import agents

st.set_page_config(page_title="Performance agents", page_icon="🧑‍💼", layout="wide")
ui.inject_css()
ui.hero("Performance des agents",
        "Évaluation individuelle, segmentation et facteurs de performance",
        icon="🧑‍💼", theme="violet")

df = get_data()

min_calls = st.slider("Seuil minimal d'appels traités (filtre les agents marginaux)",
                      50, 1000, 100, step=50)
ag = agents.agent_kpis(df, min_calls=min_calls)
seg = agents.segment_agents(ag, k=3)

ui.kpi_row([
    {"icon": "🧑‍💼", "value": fmt_int(len(seg)), "label": "Agents évalués", "color": "#8e44ad"},
    {"icon": "⏱️", "value": f"{ag['aht_moyen'].mean():.0f} s", "label": "AHT moyen (global)", "color": "#e67e22"},
    {"icon": "⚡", "value": f"{ag['productivite_jour'].mean():.0f}", "label": "Productivité moy. / jour", "color": "#2ecc71"},
    {"icon": "📞", "value": fmt_int(ag['appels_traites'].sum()), "label": "Appels traités (total)", "color": "#1f4e79"},
])

st.write("")

# --- Segmentation (scatter) ------------------------------------------------
ui.section("🎯 Segmentation des agents (K-means)")
st.caption("Axe X : productivité quotidienne — Axe Y : AHT. "
           "Profil idéal = haute productivité, AHT maîtrisé.")
fig = px.scatter(
    seg, x="productivite_jour", y="aht_moyen", size="appels_traites",
    color="segment", hover_name="server",
    labels={"productivite_jour": "Productivité (appels/jour)",
            "aht_moyen": "AHT moyen (s)", "segment": "Segment"},
    color_discrete_sequence=[PALETTE["ok"], PALETTE["primary"], PALETTE["warn"]])
fig.update_layout(height=460)
st.plotly_chart(fig, use_container_width=True)

# --- Classements -----------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    ui.section("🏆 Top 10 — volume traité")
    top = seg.nlargest(10, "appels_traites")
    figt = px.bar(top, x="appels_traites", y="server", orientation="h",
                  color="segment", labels={"appels_traites": "Appels", "server": ""},
                  color_discrete_sequence=[PALETTE["ok"], PALETTE["primary"], PALETTE["warn"]])
    figt.update_layout(height=400, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(figt, use_container_width=True)

with col2:
    ui.section("⏱️ AHT par agent (10 plus élevés)")
    high_aht = seg.nlargest(10, "aht_moyen")
    figa = px.bar(high_aht, x="aht_moyen", y="server", orientation="h",
                  labels={"aht_moyen": "AHT (s)", "server": ""})
    figa.update_traces(marker_color=PALETTE["accent"])
    figa.update_layout(height=400, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(figa, use_container_width=True)

# --- Facteur : type de service --------------------------------------------
st.divider()
ui.section("🔧 Facteur explicatif — AHT par type de service")
st.caption("Les types Support Internet et Bourse exigent des durées de traitement contrastées : "
           "à intégrer dans l'affectation par compétences.")
pt = agents.perf_by_type(df)
figp = px.bar(pt, x="type_label", y="aht_moyen", text="aht_moyen",
              labels={"type_label": "Type de service", "aht_moyen": "AHT moyen (s)"})
figp.update_traces(marker_color=PALETTE["primary"])
st.plotly_chart(figp, use_container_width=True)

# --- Table détaillée -------------------------------------------------------
ui.section("📋 Détail par agent")
st.dataframe(
    seg.rename(columns={
        "server": "Agent", "appels_traites": "Appels", "aht_moyen": "AHT moy (s)",
        "aht_median": "AHT méd (s)", "jours_actifs": "Jours actifs",
        "productivite_jour": "Prod./jour", "attente_moyenne_s": "Attente moy (s)",
        "segment": "Segment"}),
    use_container_width=True, hide_index=True)
