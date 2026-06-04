# Application BI & Machine Learning — Centre d'appel AnonymousBank (1999)

Dispositif décisionnel complet pour l'**optimisation de la planification des effectifs**
d'un centre de relation client bancaire : Business Intelligence + modèles prédictifs.

## Démarrage rapide

```powershell
# Depuis le dossier app\
pip install -r requirements.txt        # une seule fois
python -m streamlit run app.py         # ou : .\run.ps1
```

L'application s'ouvre dans le navigateur (http://localhost:8501).
Le premier chargement nettoie les 444 000 appels et crée un cache (`_cache/calls_clean.parquet`) ;
les lancements suivants sont instantanés.

## Architecture

```
app/
├── app.py                     # Page d'accueil (vue d'ensemble, qualité des données)
├── pages/
│   ├── 1_Vue_executive.py     # KPI globaux, heatmap de charge, profils
│   ├── 2_Previsions_volume.py # SARIMA / Prophet / naïf — prévision quotidienne
│   ├── 3_Performance_agents.py# KPI agents + segmentation K-means
│   ├── 4_Risque_abandon.py    # 4 classifieurs comparés + ROC + importances
│   ├── 5_Dimensionnement.py   # Erlang C / A → agents requis par créneau
│   └── 6_Survie_patience.py   # Kaplan-Meier + Cox de la patience client
├── src/
│   ├── config.py              # chemins, libellés métier, constantes
│   ├── etl.py                 # extraction, nettoyage, feature engineering
│   ├── kpi.py                 # indicateurs de pilotage
│   ├── forecasting.py         # séries temporelles (naïf, SARIMA, Prophet, LSTM)
│   ├── abandonment.py         # classification de l'abandon + SHAP
│   ├── agents.py              # performance & segmentation des agents
│   ├── staffing.py            # dimensionnement Erlang
│   ├── survival.py            # analyse de survie de la patience
│   └── ui.py                  # helpers Streamlit (cache, thème)
├── requirements.txt
└── run.ps1
```

## Modules analytiques

| Objectif | Méthode | Page |
|----------|---------|------|
| Prévision du volume | Naïf saisonnier, SARIMA, **Prophet**, **LSTM** (TensorFlow) | Prévisions |
| Performance agents | KPI (AHT, productivité) + K-means | Performance |
| Prédiction d'abandon | Régression logistique, Random Forest, Gradient Boosting, XGBoost + **interprétabilité SHAP** | Risque d'abandon |
| Dimensionnement | Erlang C + correction Erlang A | Dimensionnement |
| Patience client | **Kaplan-Meier + Cox** (analyse de survie) | Survie / patience |

> **Prophet** : la librairie est intégrée, mais son backend de compilation (Stan) peut
> échouer sur certaines configurations Windows/Python 3.8 (`signal 0xC0000139` = conflit de
> DLL). L'application le détecte, l'affiche clairement et continue avec les autres modèles
> (repli SARIMA). Le benchmark mesuré ici donne : **SARIMA ≈ 12,6 % MAPE > naïf ≈ 14,9 % >
> LSTM ≈ 16,7 %** — résultat conforme à la littérature (les méthodes statistiques surpassent
> souvent le deep learning sur des séries courtes, cf. Makridakis et al., compétition M4).

## Données

Source : `../Annee1999.csv` (TSV, ~444 000 appels, projet DataMOCCA / Technion).
Le chemin est configurable dans `src/config.py` (`DATA_RAW`).

## Notes méthodologiques

- **Anti-fuite** (abandon) : le temps d'attente total n'est pas utilisé comme variable
  (il constitue la cible pour les abandons). Les prédicteurs sont disponibles à l'arrivée.
- **Interprétabilité (SHAP)** : valeurs de Shapley (Lundberg & Lee, 2017) sur les modèles
  d'arbres — importance globale, beeswarm (impact + direction), dépendance et décomposition
  individuelle (waterfall) d'une prédiction d'abandon.
- **Validation temporelle** (prévision) : hold-out chronologique, pas d'aléatoire.
- **Déséquilibre de classe** : `class_weight='balanced'` / pondération XGBoost.
- **Prophet** : si non installé, repli automatique sur SARIMA.
