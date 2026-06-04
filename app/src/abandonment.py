"""PrEdiction de l'abandon d'appel : classification supervisEe comparant 4 algorithmes.

Cible : abandoned (1 = HANG, 0 = AGENT) sur les appels rEels mis en file d'attente.
Pour Eviter toute fuite d'information (data leakage), on n'utilise PAS le temps d'attente
total (q_time), qui est, pour un abandon, la patience effective du client. On s'appuie sur
des variables connues A l'arrivEe : crEneau horaire, jour, type, prioritE, charge offerte,
duree de navigation VRU.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

NUM_FEATURES = ["hour", "weekday", "month", "is_weekend", "offered_load_30min", "vru_time"]
CAT_FEATURES = ["type", "priority"]
TARGET = "abandoned"


def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Restreint aux appels rEels entrEs en file et prepare les colonnes du modEle."""
    data = df[(df["phantom"] == 0) & (df["entered_queue"] == 1)].copy()
    cols = NUM_FEATURES + CAT_FEATURES + [TARGET]
    data = data[cols].dropna(subset=NUM_FEATURES + [TARGET])
    data["priority"] = data["priority"].astype(str)
    data["type"] = data["type"].astype(str)
    return data


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])


def _models() -> dict:
    models = {
        "Regression logistique": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, class_weight="balanced",
            n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    if _HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            n_jobs=-1, random_state=42)
    return models


def _evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "AUC-ROC": round(roc_auc_score(y_true, y_proba), 4),
    }


def train_and_compare(df: pd.DataFrame, test_size: float = 0.25, sample: int | None = 120000):
    """Entraine et compare les modEles. Echantillonne pour la rEactivitE de l'app.

    Renvoie : (table_comparative, dict_resultats_detailles, infos).
    """
    data = build_dataset(df)
    if sample and len(data) > sample:
        data = data.sample(sample, random_state=42)

    X = data[NUM_FEATURES + CAT_FEATURES]
    y = data[TARGET].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42)

    pre = _preprocessor()
    rows, detailed = [], {}
    for name, clf in _models().items():
        pipe = Pipeline([("prep", pre), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics = _evaluate(y_te, pred, proba)
        fpr, tpr, _ = roc_curve(y_te, proba)
        cm = confusion_matrix(y_te, pred)
        rows.append({"Modele": name, **metrics})
        detailed[name] = {"metrics": metrics, "fpr": fpr, "tpr": tpr,
                          "cm": cm, "pipe": pipe}

    table = pd.DataFrame(rows).sort_values("AUC-ROC", ascending=False).reset_index(drop=True)
    infos = {
        "n_appels_file": int(len(build_dataset(df))),
        "n_entraine": int(len(data)),
        "taux_abandon_pct": round(100 * y.mean(), 2),
        "xgboost_dispo": _HAS_XGB,
    }
    return table, detailed, infos


def feature_importance(detailed: dict, model_name: str = "Random Forest") -> pd.DataFrame:
    """Importance des variables pour un modEle A base d'arbres."""
    if model_name not in detailed:
        return pd.DataFrame()
    pipe = detailed[model_name]["pipe"]
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return pd.DataFrame()
    ohe = pipe.named_steps["prep"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CAT_FEATURES))
    names = NUM_FEATURES + cat_names
    imp = pd.DataFrame({"variable": names, "importance": clf.feature_importances_})
    return imp.sort_values("importance", ascending=False).reset_index(drop=True)


# --- Interpretabilite SHAP -------------------------------------------------
try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False


def has_shap() -> bool:
    return _HAS_SHAP


# Libelles lisibles des variables (pour les graphiques SHAP)
FEATURE_LABELS = {
    "hour": "Heure d'arrivée", "weekday": "Jour de semaine", "month": "Mois",
    "is_weekend": "Week-end", "offered_load_30min": "Charge offerte (30 min)",
    "vru_time": "Durée navigation VRU",
    "type_PS": "Type: Service courant", "type_PE": "Type: Anglophone",
    "type_NW": "Type: Prospect", "type_NE": "Type: NE",
    "type_IN": "Type: Internet", "type_TT": "Type: Bourse", "type_AA": "Type: Autre",
    "priority_0": "Priorité standard", "priority_1": "Priorité 1",
    "priority_2": "Priorité haute",
}


def pretty(name: str) -> str:
    return FEATURE_LABELS.get(name, name)


def compute_shap(pipe, df, sample: int = 2000, seed: int = 42) -> dict:
    """Calcule les valeurs SHAP (classe = abandon) pour un modEle A base d'arbres.

    Renvoie un dict : shap_values (n, p), X_transformed (n, p), X_raw (DataFrame),
    feature_names (list), expected_value (float, Echelle log-odds).
    """
    if not _HAS_SHAP:
        raise RuntimeError("Librairie shap indisponible")

    data = build_dataset(df)
    if len(data) > sample:
        data = data.sample(sample, random_state=seed)
    X = data[NUM_FEATURES + CAT_FEATURES].reset_index(drop=True)

    pre = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]
    Xt = pre.transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    ohe = pre.named_transformers_["cat"]
    names = NUM_FEATURES + list(ohe.get_feature_names_out(CAT_FEATURES))

    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(Xt)
    expected = explainer.expected_value

    # Normalisation des formats (XGBoost -> array ; RandomForest -> liste par classe)
    if isinstance(sv, list):
        sv = sv[1]
        expected = expected[1] if hasattr(expected, "__len__") else expected
    elif getattr(sv, "ndim", 2) == 3:
        sv = sv[:, :, 1]
        expected = expected[1] if hasattr(expected, "__len__") else expected
    expected = float(np.ravel(expected)[0]) if hasattr(expected, "__len__") else float(expected)

    return {
        "shap_values": np.asarray(sv), "X_transformed": np.asarray(Xt),
        "X_raw": X, "feature_names": names, "expected_value": expected,
    }


def shap_global_importance(shap_res: dict) -> pd.DataFrame:
    """Importance globale = moyenne des |valeurs SHAP| par variable."""
    sv = shap_res["shap_values"]
    names = shap_res["feature_names"]
    mean_abs = np.abs(sv).mean(axis=0)
    out = pd.DataFrame({"variable": names, "label": [pretty(n) for n in names],
                        "importance_shap": mean_abs})
    return out.sort_values("importance_shap", ascending=False).reset_index(drop=True)
