"""Helpers partagés pour l'interface Streamlit : chargement en cache, thème, composants."""
from __future__ import annotations
import os
import sys

# Permet aux pages d'importer le package `src`
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import streamlit as st
from src import etl


PALETTE = {
    "primary": "#1f4e79", "accent": "#e67e22", "ok": "#2ecc71",
    "warn": "#e74c3c", "muted": "#7f8c8d",
}

_CSS = """
<style>
  .block-container { padding-top: 2rem; max-width: 1200px; }
  /* En-tête (hero) */
  .hero { padding: 1.9rem 2.3rem; border-radius: 18px; color: #fff;
    box-shadow: 0 10px 30px rgba(20,58,92,.26); margin-bottom: 1.4rem; }
  .hero h1 { color:#fff; font-size:2.0rem; font-weight:800; margin:0 0 .3rem 0; line-height:1.15; }
  .hero .sub { font-size:1.0rem; opacity:.96; margin:.1rem 0; font-weight:500; }
  .hero .corpus { font-size:.84rem; opacity:.82; margin-top:.6rem;
    border-top:1px solid rgba(255,255,255,.25); padding-top:.55rem; display:inline-block; }
  /* Cartes KPI */
  .kpi { background:#fff; border-radius:15px; padding:1.05rem 1.2rem; height:100%;
    border:1px solid #edf0f4; box-shadow:0 3px 14px rgba(0,0,0,.05);
    border-top:4px solid var(--c,#1f4e79); transition:transform .15s ease; }
  .kpi:hover { transform:translateY(-3px); box-shadow:0 8px 22px rgba(0,0,0,.10); }
  .kpi .ico { font-size:1.4rem; }
  .kpi .val { font-size:1.8rem; font-weight:800; color:#172033; line-height:1.05; margin-top:.2rem; }
  .kpi .lab { font-size:.73rem; color:#6b7280; text-transform:uppercase; letter-spacing:.05em;
    margin-top:.3rem; font-weight:600; }
  .kpi .sublab { font-size:.74rem; margin-top:.3rem; font-weight:600; }
  /* Pipeline */
  .pipe { display:flex; flex-wrap:wrap; align-items:center; gap:.4rem; margin:.4rem 0 .2rem 0; }
  .pipe .step { background:#eef4fb; color:#1f4e79; border:1px solid #d4e3f3;
    padding:.42rem .8rem; border-radius:9px; font-weight:600; font-size:.86rem; }
  .pipe .arr { color:#9aa7b4; font-weight:700; }
  /* Cartes module */
  .mod { background:#fff; border:1px solid #edf0f4; border-left:4px solid var(--c,#1f4e79);
    border-radius:12px; padding:.85rem 1rem; box-shadow:0 2px 8px rgba(0,0,0,.04); height:100%; }
  .mod .t { font-weight:700; color:#172033; font-size:.96rem; }
  .mod .d { color:#6b7280; font-size:.82rem; margin-top:.2rem; }
  /* Chips */
  .chip { background:#f6f9fc; border:1px solid #e4ecf4; border-radius:11px;
    padding:.7rem .9rem; text-align:center; }
  .chip .v { font-size:1.35rem; font-weight:800; color:#1f4e79; }
  .chip .l { font-size:.72rem; color:#6b7280; text-transform:uppercase; letter-spacing:.04em; margin-top:.2rem; }
  .sec-title { font-size:1.22rem; font-weight:700; color:#172033; margin:.4rem 0 .7rem 0; }
</style>
"""

# Dégradés d'en-tête par thème de page
GRADIENTS = {
    "navy": ("#143a5c", "#2e74b5"), "violet": ("#4a235a", "#8e44ad"),
    "red": ("#7b241c", "#e74c3c"), "orange": ("#7e3f0a", "#e67e22"),
    "teal": ("#0b5345", "#16a085"), "blue": ("#1a5276", "#2e86c1"),
}


@st.cache_data(show_spinner="Chargement et nettoyage des données...")
def get_data():
    """Charge la table propre (cache parquet) une seule fois par session."""
    return etl.load_clean()


def inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str = "", icon: str = "📊", theme: str = "navy", corpus: str = ""):
    c1, c2 = GRADIENTS.get(theme, GRADIENTS["navy"])
    extra = f'<div class="corpus">{corpus}</div>' if corpus else ""
    st.markdown(
        f"""<div class="hero" style="background:linear-gradient(135deg,{c1} 0%,{c2} 100%)">
              <h1>{icon} {title}</h1>
              <div class="sub">{subtitle}</div>{extra}
            </div>""",
        unsafe_allow_html=True,
    )


def kpi_row(items):
    """items : liste de dicts {icon, value, label, color, sub?, sub_color?}."""
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        sub = ""
        if it.get("sub"):
            sub = f'<div class="sublab" style="color:{it.get("sub_color", "#6b7280")}">{it["sub"]}</div>'
        col.markdown(
            f"""<div class="kpi" style="--c:{it['color']}">
                  <div class="ico">{it.get('icon', '')}</div>
                  <div class="val">{it['value']}</div>
                  <div class="lab">{it['label']}</div>{sub}
                </div>""",
            unsafe_allow_html=True,
        )


def section(title: str):
    st.markdown(f'<div class="sec-title">{title}</div>', unsafe_allow_html=True)


def fmt_int(x) -> str:
    try:
        return f"{int(x):,}".replace(",", " ")
    except Exception:
        return str(x)


def fmt_sec(x) -> str:
    try:
        x = float(x)
        m, s = divmod(int(round(x)), 60)
        return f"{m}m {s:02d}s" if m else f"{s}s"
    except Exception:
        return str(x)
