import streamlit as st
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

st.set_page_config(
    page_title="EBT · Pronósticos",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from dashboard import render_prediction_card

if hasattr(st, "navigation") and hasattr(st, "Page"):
    navegacion = st.navigation(
        [st.Page(render_prediction_card, title="Prediction Card", icon="⚽")],
        position="hidden",
    )
    navegacion.run()
else:
    # Compatibilidad con versiones anteriores: las páginas legacy se muestran
    # como vistas desactivadas, sin lógica ni componentes financieros.
    render_prediction_card()
