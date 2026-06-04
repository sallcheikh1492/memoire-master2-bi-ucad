"""Page 4 — Prédiction du risque d'abandon : comparaison de 4 classifieurs + SHAP."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src import ui
from src.ui import get_data, fmt_int, PALETTE
from src import abandonment

st.set_page_config(page_title="Risque d'abandon", page_icon="🚨", layout="wide")
ui.inject_css()
ui.hero("Prédiction du risque d'abandon",
        "Classification supervisée — Régression logistique · Random Forest · Gradient Boosting · XGBoost",
        icon="🚨", theme="red")

df = get_data()

st.info(
    "Pour éviter toute fuite d'information, le modèle n'utilise PAS le temps d'attente total "
    "(qui, pour un abandon, EST la patience du client). Variables : créneau horaire, jour, mois, "
    "week-end, type, priorité, charge offerte (congestion), durée de navigation VRU."
)

sample = st.select_slider(
    "Taille de l'échantillon d'entraînement (réactivité vs précision)",
    options=[30000, 60000, 100000, 150000], value=60000)


@st.cache_data(show_spinner="Entraînement et comparaison des 4 modèles...")
def _train(sample):
    return abandonment.train_and_compare(df, sample=sample)


table, detailed, infos = _train(sample)

ui.kpi_row([
    {"icon": "📥", "value": fmt_int(infos['n_appels_file']), "label": "Appels en file (population)", "color": "#1f4e79"},
    {"icon": "🎓", "value": fmt_int(infos['n_entraine']), "label": "Échantillon entraîné", "color": "#2e74b5"},
    {"icon": "📉", "value": f"{infos['taux_abandon_pct']} %", "label": "Taux d'abandon (classe +)", "color": "#e74c3c"},
])

st.write("")

# --- Tableau comparatif ----------------------------------------------------
ui.section("📊 Tableau comparatif des modèles")
st.caption("Pour l'abandon, privilégier le **Recall** (ne pas rater un client à risque) "
           "et l'**AUC-ROC**.")
st.dataframe(
    table, use_container_width=True, hide_index=True,
    column_config={
        "AUC-ROC": st.column_config.ProgressColumn(
            "AUC-ROC", min_value=0.0, max_value=1.0, format="%.3f"),
        "Recall": st.column_config.ProgressColumn(
            "Recall", min_value=0.0, max_value=1.0, format="%.3f"),
        "F1": st.column_config.ProgressColumn(
            "F1", min_value=0.0, max_value=1.0, format="%.3f"),
    })
best = table.iloc[0]["Modele"]
st.success(f"✅ Meilleur AUC-ROC : **{best}** ({table.iloc[0]['AUC-ROC']})")

# --- Courbes ROC -----------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    ui.section("📈 Courbes ROC")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash", color="gray"), name="Hasard"))
    palette = [PALETTE["primary"], PALETTE["accent"], PALETTE["ok"], PALETTE["warn"]]
    for i, (name, d) in enumerate(detailed.items()):
        fig.add_trace(go.Scatter(x=d["fpr"], y=d["tpr"], name=name,
                                 line=dict(color=palette[i % 4])))
    fig.update_layout(height=420, xaxis_title="Taux de faux positifs",
                      yaxis_title="Taux de vrais positifs (Recall)")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    ui.section("🔑 Importance des variables")
    st.caption("Modèle Random Forest — facteurs explicatifs de l'abandon.")
    imp = abandonment.feature_importance(detailed, "Random Forest")
    if not imp.empty:
        figi = px.bar(imp.head(10), x="importance", y="variable", orientation="h",
                      labels={"importance": "Importance", "variable": ""})
        figi.update_traces(marker_color=PALETTE["primary"])
        figi.update_layout(height=420, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(figi, use_container_width=True)

# --- Matrice de confusion du meilleur modèle -------------------------------
st.divider()
ui.section(f"🔢 Matrice de confusion — {best}")
cm = detailed[best]["cm"]
cm_df = pd.DataFrame(cm, index=["Réel: servi", "Réel: abandon"],
                     columns=["Prédit: servi", "Prédit: abandon"])
figcm = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues", aspect="auto")
figcm.update_layout(height=360)
st.plotly_chart(figcm, use_container_width=True)

st.caption("Usage opérationnel : un appel dont la probabilité prédite d'abandon dépasse un seuil "
           "calibré déclenche une alerte (renfort agent ou rappel automatique / *virtual queue*).")

# ===========================================================================
# INTERPRÉTABILITÉ SHAP
# ===========================================================================
st.divider()
ui.section("🔍 Interprétabilité du modèle (SHAP)")
st.caption("Les valeurs de Shapley (Lundberg & Lee, 2017) décomposent chaque prédiction "
           "en contributions additives des variables — IA explicable.")

if not abandonment.has_shap():
    st.info("ℹ️ L'analyse d'interprétabilité **SHAP** (beeswarm, dépendance, cascade) est "
            "disponible dans la version locale du projet. Elle est désactivée sur la démo en "
            "ligne pour un chargement plus léger et rapide.")
    st.stop()

tree_models = [m for m in detailed if m in ("XGBoost", "Random Forest", "Gradient Boosting")]
cs1, cs2 = st.columns([2, 1])
shap_model = cs1.selectbox("Modèle à expliquer (modèles à base d'arbres)", tree_models,
                           index=0 if "XGBoost" in tree_models else 0)
shap_n = cs2.select_slider("Échantillon expliqué", options=[1000, 2000, 3000], value=2000)


@st.cache_data(show_spinner="Calcul des valeurs SHAP...")
def _shap(_pipe, model_name, n):
    return abandonment.compute_shap(_pipe, df, sample=n)


shap_res = _shap(detailed[shap_model]["pipe"], shap_model, shap_n)
sv = shap_res["shap_values"]
names = shap_res["feature_names"]
labels = [abandonment.pretty(n) for n in names]
Xt = shap_res["X_transformed"]

st.info(
    "**Lecture** : une valeur SHAP positive pousse la prédiction *vers l'abandon* ; "
    "négative, *vers l'appel servi*. Les valeurs sont en échelle log-odds. "
    f"Base (log-odds moyen) = {shap_res['expected_value']:.3f}."
)

# --- Importance globale + Beeswarm ----------------------------------------
g1, g2 = st.columns(2)
with g1:
    ui.section("Importance globale")
    st.caption("Moyenne des |valeurs SHAP| — impact moyen de chaque variable.")
    imp = abandonment.shap_global_importance(shap_res).head(12)
    figg = px.bar(imp, x="importance_shap", y="label", orientation="h",
                  labels={"importance_shap": "Impact SHAP moyen (|valeur|)", "label": ""})
    figg.update_traces(marker_color=PALETTE["primary"])
    figg.update_layout(height=460, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(figg, use_container_width=True)

with g2:
    ui.section("Beeswarm (impact + direction)")
    st.caption("Chaque point = un appel. Couleur = valeur de la variable (rouge=élevée, bleu=faible).")
    order = np.argsort(np.abs(sv).mean(axis=0))[::-1][:10]
    rng = np.random.default_rng(0)
    figb = go.Figure()
    for pos, idx in enumerate(order[::-1]):
        vals = sv[:, idx]
        fv = Xt[:, idx].astype(float)
        rngv = fv.max() - fv.min()
        norm = (fv - fv.min()) / rngv if rngv > 0 else np.zeros_like(fv)
        jitter = pos + rng.uniform(-0.32, 0.32, size=len(vals))
        figb.add_trace(go.Scatter(
            x=vals, y=jitter, mode="markers",
            marker=dict(color=norm, colorscale="RdBu_r", coloraxis="coloraxis", size=5,
                        opacity=0.6),
            name=labels[idx], hovertext=labels[idx], showlegend=False))
    figb.add_vline(x=0, line_dash="dash", line_color="gray")
    figb.update_layout(
        height=460,
        coloraxis=dict(colorscale="RdBu_r",
                       colorbar=dict(title="Valeur", tickvals=[0, 1],
                                     ticktext=["Faible", "Élevée"])),
        xaxis_title="Valeur SHAP (→ abandon)",
        yaxis=dict(tickmode="array", tickvals=list(range(len(order))),
                   ticktext=[labels[i] for i in order[::-1]]))
    st.plotly_chart(figb, use_container_width=True)

# --- Dependence plot -------------------------------------------------------
ui.section("📐 Graphique de dépendance")
st.caption("Relation entre la valeur (brute) d'une variable et son effet SHAP sur l'abandon.")
num_feats = abandonment.NUM_FEATURES
dep_feat = st.selectbox("Variable", num_feats,
                        format_func=abandonment.pretty,
                        index=num_feats.index("offered_load_30min"))
fi = names.index(dep_feat)
raw_vals = shap_res["X_raw"][dep_feat].values.astype(float)
figd = px.scatter(
    x=raw_vals, y=sv[:, fi], opacity=0.5,
    labels={"x": abandonment.pretty(dep_feat) + " (valeur observée)",
            "y": "Valeur SHAP (→ abandon)"},
    trendline="lowess" if len(raw_vals) <= 3000 else None,
    color_discrete_sequence=[PALETTE["accent"]])
figd.add_hline(y=0, line_dash="dash", line_color="gray")
figd.update_layout(height=380)
st.plotly_chart(figd, use_container_width=True)

# --- Explication individuelle (waterfall) ----------------------------------
ui.section("💧 Explication d'un appel individuel")
st.caption("Décomposition de la prédiction d'un appel : contribution de chaque variable.")
idx_call = st.slider("Indice de l'appel dans l'échantillon", 0, len(sv) - 1, 0)
contrib = pd.DataFrame({
    "variable": labels, "shap": sv[idx_call],
    "valeur": [shap_res["X_raw"].iloc[idx_call][n] if n in shap_res["X_raw"].columns else ""
               for n in names],
})
contrib["abs"] = contrib["shap"].abs()
top = contrib.sort_values("abs", ascending=False).head(8).sort_values("shap")
base = shap_res["expected_value"]
logit = base + sv[idx_call].sum()
proba = 1 / (1 + np.exp(-logit))
cc1, cc2 = st.columns([1, 2])
cc1.metric("Probabilité d'abandon prédite", f"{proba*100:.1f} %")
cc1.caption(f"Base (log-odds) {base:.2f} + contributions = {logit:.2f}")
figw = go.Figure(go.Waterfall(
    orientation="v", measure=["relative"] * len(top),
    x=top["variable"], y=top["shap"],
    decreasing={"marker": {"color": PALETTE["ok"]}},
    increasing={"marker": {"color": PALETTE["warn"]}},
    connector={"line": {"color": "rgba(150,150,150,0.4)"}}))
figw.update_layout(height=380, yaxis_title="Contribution SHAP (log-odds)",
                   title="Top 8 contributions (vert = vers servi, rouge = vers abandon)")
cc2.plotly_chart(figw, use_container_width=True)

st.success(
    "**Apport scientifique** : SHAP transforme la boîte noire (XGBoost) en recommandations "
    "actionnables — on identifie *pourquoi* un appel est à risque (ex. forte congestion, "
    "créneau, type prospect) et on cible l'action préventive."
)
