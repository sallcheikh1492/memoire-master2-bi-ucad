"""GEnEre les figures (PNG) et les chiffres (results.json) du rapport, A partir des donnEes rEelles."""
import os, sys, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import etl, kpi, forecasting, abandonment, agents, staffing, survival, config

OUT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})
C = {"p": "#1f4e79", "a": "#e67e22", "ok": "#2ecc71", "w": "#e74c3c", "m": "#7f8c8d"}
R = {}

print("Chargement..."); df = etl.load_clean()
g = kpi.global_kpis(df); R["global"] = g
print("KPI globaux OK")

# 1. Volume journalier
dv = kpi.daily_volume(df)
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(dv["date_only"], dv["volume"], color=C["p"], lw=1)
ax.axhline(dv["volume"].mean(), ls="--", color=C["a"], label=f"Moyenne = {dv['volume'].mean():.0f}")
ax.set_title("Volume d'appels quotidien (1999)"); ax.set_xlabel("Date"); ax.set_ylabel("Appels")
ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(FIG, "f1_volume.png")); plt.close(fig)

# 2. Heatmap charge
hm = kpi.heatmap_load(df)
fig, ax = plt.subplots(figsize=(8, 3.2))
im = ax.imshow(hm.values, aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(len(hm.columns))); ax.set_xticklabels(hm.columns, fontsize=7)
ax.set_yticks(range(len(hm.index))); ax.set_yticklabels(hm.index, fontsize=8)
ax.set_title("Charge moyenne — jour de semaine x heure"); ax.set_xlabel("Heure")
fig.colorbar(im, ax=ax, label="Appels"); fig.tight_layout()
fig.savefig(os.path.join(FIG, "f2_heatmap.png")); plt.close(fig)

# 3. Abandon par tranche d'attente
ab = kpi.abandonment_by_wait_bucket(df)
fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(ab["bucket"].astype(str), ab["taux_abandon"], color=C["w"])
ax.set_title("Taux d'abandon selon le temps d'attente"); ax.set_ylabel("Abandon %"); ax.set_xlabel("Attente")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "f3_abandon_wait.png")); plt.close(fig)
R["kpi_type"] = kpi.kpis_by_type(df).to_dict("records")

# 4. PrEvision : SARIMA + naif + LSTM
print("PrEvision (SARIMA + LSTM)...")
s = kpi.daily_volume(df).set_index("date_only")["volume"]
res = forecasting.run_all_models(s, test_days=21, use_lstm=True)
ti, tv = res["_test_index"], res["_test_values"]; tr = res["_train"]
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.plot(tr.index[-40:], tr.values[-40:], color=C["m"], lw=1, label="Historique")
ax.plot(ti, tv, color="black", lw=2.2, label="REel")
cols = [C["p"], C["a"], C["ok"], C["w"]]
fc_rows = []
for i, (name, r) in enumerate(res["models"].items()):
    if "forecast" in r:
        ax.plot(ti, r["forecast"], ls="--", color=cols[i % 4], label=name)
        fc_rows.append({"modele": name, **r["metrics"]})
ax.set_title("PrEvision du volume — rEel vs prEvu (hold-out)"); ax.set_ylabel("Appels"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "f4_forecast.png")); plt.close(fig)
R["forecast"] = fc_rows

# 5. Abandon : ROC 4 modEles
print("Classification abandon...")
table, detailed, infos = abandonment.train_and_compare(df, sample=80000)
R["abandon_table"] = table.to_dict("records"); R["abandon_infos"] = infos
fig, ax = plt.subplots(figsize=(5.2, 4.2))
ax.plot([0, 1], [0, 1], ls="--", color="gray")
for i, (name, d) in enumerate(detailed.items()):
    ax.plot(d["fpr"], d["tpr"], color=cols[i % 4],
            label=f"{name} (AUC={d['metrics']['AUC-ROC']:.3f})")
ax.set_title("Courbes ROC — prEdiction d'abandon"); ax.set_xlabel("Faux positifs"); ax.set_ylabel("Vrais positifs")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(FIG, "f5_roc.png")); plt.close(fig)

# 6. SHAP importance globale
print("SHAP...")
best_model = table.iloc[0]["Modele"]
try:
    sh = abandonment.compute_shap(detailed[best_model]["pipe"], df, sample=2000)
    imp = abandonment.shap_global_importance(sh).head(10)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.barh(imp["label"][::-1], imp["importance_shap"][::-1], color=C["p"])
    ax.set_title(f"Importance SHAP des variables — {best_model}"); ax.set_xlabel("Impact SHAP moyen")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "f6_shap.png")); plt.close(fig)
    R["shap_top"] = imp.to_dict("records"); R["shap_model"] = best_model
except Exception as e:
    print("SHAP KO:", e); R["shap_top"] = []

# 7. Survie KM par type
print("Survie (KM + Cox)...")
d = survival.build_survival_data(df)
R["survie"] = survival.survival_summary(d)
curves, medians, sizes = survival.km_by_group(d, "type", config.TYPE_LABELS)
fig, ax = plt.subplots(figsize=(7, 3.6))
import itertools
pal = itertools.cycle([C["p"], C["a"], C["ok"], C["w"], C["m"], "#9b59b6"])
for name, cv in curves.items():
    cv = cv[cv["t"] <= 600]
    ax.plot(cv["t"], cv["survie"], label=f"{name} (méd. {medians[name]:.0f}s)", color=next(pal))
ax.axhline(0.5, ls=":", color="gray"); ax.set_ylim(0, 1)
ax.set_title("Survie de la patience par type de service (Kaplan-Meier)")
ax.set_xlabel("Temps d'attente (s)"); ax.set_ylabel("P(patienter > t)"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "f7_km.png")); plt.close(fig)
R["km_medians"] = {k: round(v, 1) for k, v in medians.items()}

# 8. Cox HR forest plot
res_cox, mcox = survival.cox_model(d, sample=60000)
R["cox"] = res_cox.to_dict("records"); R["cox_metrics"] = mcox
pdf = res_cox[res_cox["variable"] != "Intercept"]
fig, ax = plt.subplots(figsize=(7, 3.8))
ypos = range(len(pdf))
ax.errorbar(pdf["HR"], ypos, xerr=[pdf["HR"] - pdf["HR_bas"], pdf["HR_haut"] - pdf["HR"]],
            fmt="o", color=C["p"], ecolor=C["m"], capsize=3)
ax.axvline(1.0, ls="--", color=C["w"]); ax.set_xscale("log")
ax.set_yticks(list(ypos)); ax.set_yticklabels(pdf["variable"], fontsize=8)
ax.set_title("Facteurs de risque d'abandon — Cox (Hazard Ratios)"); ax.set_xlabel("HR (échelle log)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "f8_cox.png")); plt.close(fig)

# 9. Dimensionnement
print("Dimensionnement...")
ip = kpi.intraday_profile(df).set_index("slot_label")["volume_moyen"]
aht = float(df.loc[df["served"] == 1, "aht"].mean())
plan = staffing.staffing_plan(ip.to_dict(), aht_s=aht, target_sl=0.8, target_s=20)
import pandas as pd
plan_df = pd.DataFrame(plan).sort_values("creneau")
plan_df["erlA"] = plan_df["agents_requis"].apply(lambda n: staffing.erlang_a_adjust(n, df["abandoned"].mean()))
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.bar(plan_df["creneau"], plan_df["volume_prevu"], color="#cfd6da", label="Volume")
ax2 = ax.twinx()
ax2.plot(plan_df["creneau"], plan_df["agents_requis"], color=C["w"], lw=2, label="Agents (Erlang C)")
ax2.plot(plan_df["creneau"], plan_df["erlA"], color=C["ok"], lw=2, ls=":", label="Agents (Erlang A)")
ax.set_xticklabels(plan_df["creneau"], rotation=90, fontsize=6)
ax.set_title("Plan d'effectifs par créneau (AHT=%.0fs)" % aht); ax.set_ylabel("Volume"); ax2.set_ylabel("Agents")
ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "f9_staffing.png")); plt.close(fig)
R["staffing"] = {"aht": round(aht, 1), "pic_erlangC": int(plan_df["agents_requis"].max()),
                 "pic_erlangA": int(plan_df["erlA"].max()),
                 "agents_heures": round(plan_df["agents_requis"].sum() / 2, 0)}

import math
def _clean(o):
    """Remplace les NaN par None : NaN n'est pas un JSON valide (echec cote Node sinon)."""
    if isinstance(o, float):
        return None if math.isnan(o) else o
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(x) for x in o]
    return o

with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
    json.dump(_clean(R), f, ensure_ascii=False, indent=2, default=float)
print("TERMINE — figures:", len(os.listdir(FIG)), "| results.json ecrit")
