import streamlit as st

st.set_page_config(
    page_title="EBT",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚽ EBT")
st.caption("Motor Estadístico de Fútbol")

st.divider()

st.markdown("## Bienvenido")

st.write(
    """
    Esta aplicación permite analizar partidos de fútbol mediante un modelo
    estadístico basado en Expected Goals, Poisson y Backtesting.
    """
)

st.info("Selecciona una opción en el menú lateral para comenzar.")