"""Page 1 — Vue exécutive : KPI globaux, charge, qualité de service."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src import ui
from src.ui import get_data, fmt_int, fmt_sec, PALETTE
from src import kpi, config

st.set_page_config(page_title="Vue exécutive", page_icon="📊", layout="wide")
ui.inject_css()
ui.hero("Vue exécutive", "Indicateurs clés de pilotage du centre d'appel", icon="📊", theme="blue")

df = get_data()

# --- Filtres ---------------------------------------------------------------
types = sorted(df["type_label"].dropna().unique())
sel = st.multiselect("Filtrer par type de service", types, default=types)
data = df[df["type_label"].isin(sel)] if sel else df

g = kpi.global_kpis(data)

ui.kpi_row([
    {"icon": "📞", "value": fmt_int(g["appels_total"]), "label": "Appels (hors fantômes)", "color": "#1f4e79"},
    {"icon": "✅", "value": f"{g['taux_service']} %", "label": "Taux de service", "color": "#2ecc71"},
    {"icon": "📉", "value": f"{g['taux_abandon']} %", "label": "Taux d'abandon", "color": "#e74c3c",
     "sub": f"↑ {g['taux_abandon']-15:.1f} pts vs cible 15%", "sub_color": "#e74c3c"},
    {"icon": "⏱️", "value": fmt_sec(g["aht_moyen_s"]), "label": "AHT moyen", "color": "#e67e22"},
    {"icon": "🎯", "value": f"{g['service_level_pct']} %", "label": "Niveau de service (<20s)", "color": "#8e44ad",
     "sub": f"↓ {g['service_level_pct']-80:.1f} pts vs 80%", "sub_color": "#e74c3c"},
])

if g["service_level_pct"] < config.DEFAULT_SLA_TARGET * 100:
    st.warning(
        f"⚠️ Niveau de service ({g['service_level_pct']} %) très en deçà de la cible de "
        f"{int(config.DEFAULT_SLA_TARGET*100)} %. Le sous-dimensionnement aux heures de pointe "
        "est la piste prioritaire (voir page *Dimensionnement*)."
    )

st.write("")

# --- Volume journalier -----------------------------------------------------
col1, col2 = st.columns([3, 2])
with col1:
    ui.section("📈 Volume d'appels quotidien")
    dv = kpi.daily_volume(data)
    fig = px.line(dv, x="date_only", y="volume",
                  labels={"date_only": "Date", "volume": "Appels"})
    fig.update_traces(line_color=PALETTE["primary"])
    fig.add_hline(y=dv["volume"].mean(), line_dash="dash",
                  line_color=PALETTE["accent"], annotation_text="Moyenne")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    ui.section("🍩 Répartition par type de service")
    bt = kpi.kpis_by_type(data)
    fig2 = px.pie(bt, names="type_label", values="appels", hole=0.45)
    st.plotly_chart(fig2, use_container_width=True)

# --- Heatmap de charge -----------------------------------------------------
ui.section("🔥 Heatmap de charge — jour de semaine × heure")
st.caption("Volume moyen d'appels par occurrence. Identifie pics et creux pour la planification.")
hm = kpi.heatmap_load(data)
fig3 = go.Figure(data=go.Heatmap(
    z=hm.values, x=[f"{h}h" for h in hm.columns], y=hm.index,
    colorscale="YlOrRd", colorbar=dict(title="Appels")))
fig3.update_layout(height=380)
st.plotly_chart(fig3, use_container_width=True)

# --- Profil intra-journalier + KPI par type --------------------------------
col3, col4 = st.columns(2)
with col3:
    ui.section("🕒 Profil intra-journalier (créneaux 30 min)")
    ip = kpi.intraday_profile(data)
    fig4 = px.bar(ip, x="slot_label", y="volume_moyen",
                  labels={"slot_label": "Créneau", "volume_moyen": "Appels moyens"})
    fig4.update_traces(marker_color=PALETTE["primary"])
    st.plotly_chart(fig4, use_container_width=True)

with col4:
    ui.section("📋 KPI par type de service")
    st.dataframe(
        bt.rename(columns={
            "type_label": "Type", "appels": "Appels",
            "taux_abandon": "Abandon %", "aht_moyen": "AHT (s)",
            "attente_moyenne": "Attente (s)"}),
        use_container_width=True, hide_index=True)

st.divider()
ui.section("📉 Relation attente ↔ abandon")
st.caption("Le taux d'abandon décroît globalement avec l'attente : les impatients raccrochent tôt.")
ab = kpi.abandonment_by_wait_bucket(data)
fig5 = px.bar(ab, x="bucket", y="taux_abandon",
              labels={"bucket": "Temps d'attente", "taux_abandon": "Taux d'abandon %"},
              text="taux_abandon")
fig5.update_traces(marker_color=PALETTE["warn"])
st.plotly_chart(fig5, use_container_width=True)
