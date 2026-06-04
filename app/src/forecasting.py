"""PrEvision du volume d'appels : modEle de rEfErence (naIf saisonnier) + SARIMA,
avec repli automatique. Prophet est utilisE s'il est installE, sinon ignorE.
"""
from __future__ import annotations
import warnings
import importlib.util
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    _HAS_SARIMA = True
except Exception:
    _HAS_SARIMA = False

import os as _os
try:
    from prophet import Prophet
    _PROPHET_IMPORTABLE = True
except Exception:
    _PROPHET_IMPORTABLE = False

# Prophet est dEsactivE par dEfaut : sur ce poste, son binaire Stan dEclenche une erreur
# Windows (conflit de DLL TBB : point d'entrEe introuvable) qui bloque l'application par une
# fenEtre modale. Pour le rEactiver sur un environnement sain (ex. conda-forge), dEfinir la
# variable d'environnement MEMOIRE_ENABLE_PROPHET=1.
_HAS_PROPHET = _PROPHET_IMPORTABLE and _os.environ.get("MEMOIRE_ENABLE_PROPHET", "0") == "1"

# TensorFlow est lourd A importer : on dEtecte sa prEsence sans l'importer.
_HAS_TF = importlib.util.find_spec("tensorflow") is not None


def forecasting_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    mask = y_true > 0
    mape = np.mean(np.abs(err[mask] / y_true[mask])) * 100 if mask.any() else np.nan
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE_%": round(mape, 2)}


def train_test_split_ts(series: pd.Series, test_days: int = 30):
    train = series.iloc[:-test_days]
    test = series.iloc[-test_days:]
    return train, test


def seasonal_naive(train: pd.Series, horizon: int, period: int = 7) -> np.ndarray:
    """PrEvision naIve saisonniEre : reproduit la derniEre saison observEe."""
    last_season = train.values[-period:]
    reps = int(np.ceil(horizon / period))
    return np.tile(last_season, reps)[:horizon]


def fit_sarima(train: pd.Series, horizon: int, seasonal_period: int = 7):
    """SARIMA(1,1,1)(1,1,1)_m. Renvoie (prevision, ic_bas, ic_haut)."""
    if not _HAS_SARIMA:
        raise RuntimeError("statsmodels indisponible")
    model = SARIMAX(
        train, order=(1, 1, 1), seasonal_order=(1, 1, 1, seasonal_period),
        enforce_stationarity=False, enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    fc = res.get_forecast(steps=horizon)
    mean = fc.predicted_mean.values
    ci = fc.conf_int(alpha=0.20).values
    return mean, ci[:, 0], ci[:, 1]


def fit_prophet(train: pd.Series, horizon: int):
    if not _HAS_PROPHET:
        raise RuntimeError("prophet indisponible")
    dfp = pd.DataFrame({"ds": train.index, "y": train.values})
    m = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    m.fit(dfp)
    future = m.make_future_dataframe(periods=horizon, freq="D")
    fc = m.predict(future).tail(horizon)
    return fc["yhat"].values, fc["yhat_lower"].values, fc["yhat_upper"].values


def fit_lstm(train: pd.Series, horizon: int, lookback: int = 14, epochs: int = 80):
    """REseau LSTM univariE. PrEvision multi-pas rEcursive. Renvoie un array (horizon,)."""
    if not _HAS_TF:
        raise RuntimeError("tensorflow indisponible")
    import os as _os
    _os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import numpy as _np
    from sklearn.preprocessing import MinMaxScaler
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input

    tf.random.set_seed(42)
    _np.random.seed(42)

    vals = train.values.astype("float32").reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(vals)

    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i, 0])
        y.append(scaled[i, 0])
    X = _np.array(X).reshape(-1, lookback, 1)
    y = _np.array(y)

    model = Sequential([Input((lookback, 1)),
                        LSTM(50, activation="tanh"),
                        Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=16, verbose=0, shuffle=False)

    seq = scaled[-lookback:, 0].tolist()
    preds = []
    for _ in range(horizon):
        x = _np.array(seq[-lookback:]).reshape(1, lookback, 1)
        p = float(model.predict(x, verbose=0)[0, 0])
        preds.append(p)
        seq.append(p)
    preds = scaler.inverse_transform(_np.array(preds).reshape(-1, 1)).ravel()
    return _np.clip(preds, 0, None)


def run_all_models(series: pd.Series, test_days: int = 30, seasonal_period: int = 7,
                   use_lstm: bool = True) -> dict:
    """Entraine les modEles disponibles, Evalue sur le hold-out temporel.

    Renvoie un dict : {model_name: {forecast, lower, upper, metrics}} + l'index test.
    """
    series = series.asfreq("D").fillna(0)
    train, test = train_test_split_ts(series, test_days)
    horizon = len(test)
    results = {"_test_index": test.index, "_test_values": test.values,
               "_train": train, "models": {}}

    # 1) NaIf saisonnier
    sn = seasonal_naive(train, horizon, seasonal_period)
    results["models"]["Naif saisonnier"] = {
        "forecast": sn, "lower": None, "upper": None,
        "metrics": forecasting_metrics(test.values, sn),
    }

    # 2) SARIMA
    if _HAS_SARIMA:
        try:
            m, lo, hi = fit_sarima(train, horizon, seasonal_period)
            results["models"]["SARIMA"] = {
                "forecast": m, "lower": lo, "upper": hi,
                "metrics": forecasting_metrics(test.values, m),
            }
        except Exception as e:
            results["models"]["SARIMA"] = {"error": str(e)}

    # 3) Prophet (si dispo)
    if _HAS_PROPHET:
        try:
            m, lo, hi = fit_prophet(train, horizon)
            results["models"]["Prophet"] = {
                "forecast": m, "lower": lo, "upper": hi,
                "metrics": forecasting_metrics(test.values, m),
            }
        except Exception as e:
            results["models"]["Prophet"] = {"error": str(e)}

    # 4) LSTM (si dispo)
    if _HAS_TF and use_lstm:
        try:
            m = fit_lstm(train, horizon)
            results["models"]["LSTM"] = {
                "forecast": m, "lower": None, "upper": None,
                "metrics": forecasting_metrics(test.values, m),
            }
        except Exception as e:
            results["models"]["LSTM"] = {"error": str(e)}

    return results


def future_forecast(series: pd.Series, horizon: int = 14, seasonal_period: int = 7):
    """PrEvision projetEe au-delA des donnEes (meilleur modEle disponible)."""
    series = series.asfreq("D").fillna(0)
    if _HAS_SARIMA:
        try:
            m, lo, hi = fit_sarima(series, horizon, seasonal_period)
            idx = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon)
            return pd.DataFrame({"date": idx, "prevision": m.round(0),
                                 "ic_bas": lo.round(0), "ic_haut": hi.round(0)}), "SARIMA"
        except Exception:
            pass
    sn = seasonal_naive(series, horizon, seasonal_period)
    idx = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon)
    return pd.DataFrame({"date": idx, "prevision": np.round(sn),
                         "ic_bas": np.nan, "ic_haut": np.nan}), "Naif saisonnier"


def available_models() -> dict:
    return {"SARIMA": _HAS_SARIMA, "Prophet": _HAS_PROPHET, "LSTM": _HAS_TF}
