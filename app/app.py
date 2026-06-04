"""Application BI & Machine Learning - Centre d'appel bancaire AnonymousBank (1999).

Page d'accueil : présentation, qualité des données, navigation.
Lancer avec :  streamlit run app.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from src import ui
from src.ui import get_data, fmt_int, fmt_sec
from src import kpi

st.set_page_config(page_title="Centre d'appel - BI & ML", page_icon="📞",
                   layout="wide", initial_sidebar_state="expanded")
ui.inject_css()

df = get_data()
rep = df.attrs.get("report", {})
g = kpi.global_kpis(df)

ui.hero(
    "Optimisation de la planification des effectifs",
    "Business Intelligence &amp; modèles prédictifs appliqués à un centre de relation client bancaire",
    icon="📞", theme="navy",
    corpus="Corpus : <b>AnonymousBank</b> — année 1999 &nbsp;·&nbsp; projet de mémoire Master 2",
)

ui.kpi_row([
    {"icon": "📞", "value": fmt_int(g["appels_total"]), "label": "Appels analysés", "color": "#1f4e79"},
    {"icon": "✅", "value": f"{g['taux_service']} %", "label": "Taux de service", "color": "#2ecc71"},
    {"icon": "📉", "value": f"{g['taux_abandon']} %", "label": "Taux d'abandon", "color": "#e74c3c"},
    {"icon": "⏱️", "value": fmt_sec(g["aht_moyen_s"]), "label": "AHT moyen", "color": "#e67e22"},
    {"icon": "🧑‍💼", "value": fmt_int(g["nb_agents"]), "label": "Agents", "color": "#8e44ad"},
])

st.write("")
st.divider()

left, right = st.columns([3, 2], gap="large")
with left:
    ui.section("🏗️ Architecture de la solution")
    st.markdown(
        """
        <div class="pipe">
          <span class="step">Source CSV</span><span class="arr">→</span>
          <span class="step">ETL</span><span class="arr">→</span>
          <span class="step">Entrepôt (étoile)</span><span class="arr">→</span>
          <span class="step">Data Marts</span><span class="arr">→</span>
          <span class="step">Tableaux de bord</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Chaîne décisionnelle de bout en bout, alimentée par une couche analytique (Machine Learning).")
    st.write("")
    m1, m2 = st.columns(2)
    mods = [
        (m1, "📊", "Vue exécutive", "KPI, heatmap, profils", "#2e74b5"),
        (m2, "📈", "Prévision du volume", "SARIMA · LSTM · naïf saisonnier", "#1f4e79"),
        (m1, "🧑‍💼", "Performance des agents", "KPI + segmentation K-means", "#8e44ad"),
        (m2, "🚨", "Prédiction de l'abandon", "4 classifieurs + SHAP", "#e74c3c"),
        (m1, "⏳", "Survie de la patience", "Kaplan-Meier + Cox", "#16a085"),
        (m2, "👥", "Dimensionnement", "Erlang C / A → effectifs requis", "#e67e22"),
    ]
    for col, ic, t, d, c in mods:
        col.markdown(
            f"""<div class="mod" style="--c:{c}; margin-bottom:.6rem">
                  <div class="t">{ic}&nbsp; {t}</div><div class="d">{d}</div>
                </div>""",
            unsafe_allow_html=True,
        )
    st.caption("➡️ Utilisez le menu de gauche pour naviguer entre les modules.")

with right:
    ui.section("🧹 Qualité des données (ETL)")
    q1, q2, q3 = st.columns(3)
    q1.markdown(f"""<div class="chip"><div class="v">{fmt_int(rep.get('lignes_brutes_lues',0))}</div>
                <div class="l">Lignes brutes</div></div>""", unsafe_allow_html=True)
    q2.markdown(f"""<div class="chip"><div class="v">{fmt_int(rep.get('lignes_valides',0))}</div>
                <div class="l">Lignes valides</div></div>""", unsafe_allow_html=True)
    q3.markdown(f"""<div class="chip"><div class="v">{rep.get('taux_retention',0)} %</div>
                <div class="l">Rétention</div></div>""", unsafe_allow_html=True)
    st.write("")
    st.info(
        "Les lignes mal formées (délimiteurs intégrés, colonnes décalées) et les "
        f"{fmt_int(g['appels_fantomes'])} appels fantômes sont identifiés et écartés lors du nettoyage."
    )
    st.markdown(
        f"""<div class="chip" style="text-align:left">
              <div class="l">Période couverte</div>
              <div class="v" style="font-size:1.05rem">{rep.get('periode_debut','')} → {rep.get('periode_fin','')}</div>
            </div>""",
        unsafe_allow_html=True,
    )

st.divider()
ui.section("🔎 Aperçu des données nettoyées")
preview_cols = ["arrival_dt", "type_label", "priority", "outcome_label",
                "wait_time", "aht", "slot_label", "server"]
rename = {"arrival_dt": "Horodatage", "type_label": "Type de service", "priority": "Priorité",
          "outcome_label": "Issue", "wait_time": "Attente (s)", "aht": "AHT (s)",
          "slot_label": "Créneau", "server": "Agent"}
st.dataframe(df[preview_cols].head(200).rename(columns=rename),
             use_container_width=True, height=320, hide_index=True)

st.caption("Module développé dans le cadre du mémoire — approche CRISP-DM, théorie des files "
           "d'attente (Erlang), apprentissage supervisé.")
