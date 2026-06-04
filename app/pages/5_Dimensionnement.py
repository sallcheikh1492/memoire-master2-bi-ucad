"""Page 5 — Dimensionnement des effectifs (Erlang C / Erlang A)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src import ui
from src.ui import get_data, fmt_int, PALETTE
from src import kpi, staffing, forecasting

st.set_page_config(page_title="Dimensionnement", page_icon="👥", layout="wide")
ui.inject_css()
ui.hero("Dimensionnement des effectifs",
        "Traduction de la charge prévue en nombre d'agents requis — théorie des files (Erlang)",
        icon="👥", theme="orange")

df = get_data()

# --- Paramètres de service -------------------------------------------------
ui.section("⚙️ Paramètres de niveau de service")
c1, c2, c3, c4 = st.columns(4)
target_sl = c1.slider("Objectif niveau de service (%)", 50, 95, 80, step=5) / 100
target_s = c2.slider("Délai cible de réponse (s)", 10, 60, 20, step=5)
default_aht = float(df.loc[df["served"] == 1, "aht"].mean())
aht_s = c3.number_input("AHT moyen (s)", 30, 1200, int(default_aht), step=10)
abandon_rate = c4.slider("Taux d'abandon observé (Erlang A)", 0.0, 0.4,
                         float(df["abandoned"].mean()), step=0.01)

st.divider()

# --- Source de volume : profil observé OU prévision ------------------------
ui.section("📅 Plan d'effectifs par créneau de 30 min")
source = st.radio("Source du volume", ["Profil journalier moyen observé",
                                        "Prévision (volume du prochain jour)"], horizontal=True)

ip = kpi.intraday_profile(df).set_index("slot_label")["volume_moyen"]

if source.startswith("Prévision"):
    series = kpi.daily_volume(df).set_index("date_only")["volume"]
    fc, used = forecasting.future_forecast(series, horizon=1)
    next_day_total = float(fc["prevision"].iloc[0])
    shape = ip / ip.sum()
    slot_volumes = (shape * next_day_total)
    st.caption(f"Volume total prévu (modèle {used}) : **{next_day_total:.0f} appels**, "
               "réparti selon le profil intra-journalier.")
else:
    slot_volumes = ip
    st.caption("Volume moyen observé par créneau.")

# --- Calcul du plan --------------------------------------------------------
plan = staffing.staffing_plan(
    slot_volumes.to_dict(), aht_s=aht_s, target_sl=target_sl, target_s=target_s)
plan_df = pd.DataFrame(plan).sort_values("creneau").reset_index(drop=True)
plan_df["agents_erlang_a"] = plan_df["agents_requis"].apply(
    lambda n: staffing.erlang_a_adjust(n, abandon_rate))

peak = plan_df.loc[plan_df["agents_requis"].idxmax()]
ui.kpi_row([
    {"icon": "🔴", "value": f"{int(peak['agents_requis'])}", "label": "Pic d'effectif (Erlang C)",
     "color": "#e74c3c", "sub": f"créneau {peak['creneau']}", "sub_color": "#6b7280"},
    {"icon": "🟢", "value": f"{int(plan_df['agents_erlang_a'].max())}", "label": "Pic d'effectif (Erlang A)",
     "color": "#2ecc71", "sub": "corrigé de l'impatience", "sub_color": "#6b7280"},
    {"icon": "🕐", "value": f"{plan_df['agents_requis'].sum()/2:.0f}", "label": "Agents-heures / jour (Erlang C)",
     "color": "#e67e22"},
])

st.write("")

# --- Graphique -------------------------------------------------------------
fig = go.Figure()
fig.add_trace(go.Bar(x=plan_df["creneau"], y=plan_df["volume_prevu"],
                     name="Volume prévu", marker_color="rgba(127,140,141,0.4)", yaxis="y2"))
fig.add_trace(go.Scatter(x=plan_df["creneau"], y=plan_df["agents_requis"],
                         name="Agents (Erlang C)", line=dict(color=PALETTE["warn"], width=3)))
fig.add_trace(go.Scatter(x=plan_df["creneau"], y=plan_df["agents_erlang_a"],
                         name="Agents (Erlang A)", line=dict(color=PALETTE["ok"], width=3, dash="dot")))
fig.update_layout(
    height=440, xaxis_title="Créneau horaire",
    yaxis=dict(title="Agents requis"),
    yaxis2=dict(title="Volume d'appels", overlaying="y", side="right", showgrid=False),
    legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig, use_container_width=True)

# --- Tableau ---------------------------------------------------------------
ui.section("📋 Détail du plan")
show = plan_df.rename(columns={
    "creneau": "Créneau", "volume_prevu": "Volume", "trafic_erlang": "Trafic (Erlang)",
    "agents_requis": "Agents (Erlang C)", "agents_erlang_a": "Agents (Erlang A)",
    "service_level": "Niveau service", "occupation": "Occupation"})
st.dataframe(show, use_container_width=True, hide_index=True)

csv = plan_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Télécharger le plan (CSV)", csv, "plan_effectifs.csv", "text/csv")

st.caption("Méthode : trafic A = (appels × AHT) / durée du créneau ; recherche du plus petit "
           "nombre d'agents tel que le niveau de service Erlang C atteigne la cible. Erlang A "
           "corrige à la baisse pour tenir compte des abandons.")
