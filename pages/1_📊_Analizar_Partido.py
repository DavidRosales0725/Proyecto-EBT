from pathlib import Path
import sys

import streamlit as st

# -------------------------------------------------
# Path del motor
# -------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "Src"))


from motor.decision.value_bet import evaluar_value_bet
from motor.data_loader import cargar_datos
from motor.historicos import IndiceHistorico, construir_contexto_partido
from motor.core.expected_goals import calcular_goles_esperados
from motor.core.fuerzas import (
    calcular_fuerza,
    promedio_referencia,
)
from motor.core.poisson_model import (
    calcular_probabilidades,
    calcular_matriz_marcadores,
    calcular_1x2,
)
from motor.adapters.odds_api import (
    obtener_cuotas_epl,
    normalizar_cuotas_epl,
    buscar_partido,
)

# -------------------------------------------------
# Configuración de la página
# -------------------------------------------------
st.set_page_config(
    page_title="EBT - Analizar Partido",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Analizar Partido")
st.caption("Carga una temporada y analiza un partido con el motor estadístico.")

# -------------------------------------------------
# Catálogo de temporadas / archivos
# -------------------------------------------------
LIGAS = [
    "Premier League",
    "LaLiga",
    "Serie A",
    "Bundesliga",
    "Ligue 1",
]

TEMPORADAS = [
    "2024-2025",
    "2023-2024",
]

ARCHIVOS = {
    "2024-2025": "24-25.csv",
    "2023-2024": "23-24.csv",
}

# -------------------------------------------------
# Entradas de usuario
# -------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    liga = st.selectbox("Liga", LIGAS)

with col2:
    temporada = st.selectbox("Temporada", TEMPORADAS)

# -------------------------------------------------
# Cargar datos
# -------------------------------------------------
if st.button("Cargar datos"):
    ruta = ROOT / "Data" / "raw" / liga / ARCHIVOS[temporada]

    try:
        datos = cargar_datos(str(ruta), imprimir_resumen=False)
        indice = IndiceHistorico(datos.df)

        st.session_state["datos"] = datos
        st.session_state["indice"] = indice
        st.session_state["liga"] = liga
        st.session_state["temporada"] = temporada
        st.session_state["equipos"] = sorted(datos.df["equipo_local"].unique().tolist())

        st.success("Datos cargados correctamente.")

    except Exception as e:
        st.error(f"Error al cargar datos: {e}")

# -------------------------------------------------
# Mostrar datos cargados
# -------------------------------------------------
if "datos" in st.session_state:
    datos = st.session_state["datos"]
    equipos = st.session_state.get("equipos", sorted(datos.df["equipo_local"].unique().tolist()))

    st.subheader("Resumen")
    c1, c2, c3 = st.columns(3)
    c1.metric("Partidos válidos", len(datos.df))
    c2.metric("Equipos detectados", datos.df["equipo_local"].nunique())
    c3.metric("Columnas", len(datos.df.columns))

    with st.expander("Ver muestra del DataFrame"):
        st.dataframe(datos.df.head(), use_container_width=True)

    st.divider()
    st.subheader("Seleccionar partido")

    st.subheader("Seleccionar equipos")

    equipos_ordenados = equipos

    equipo_local = st.selectbox(
        "Equipo local",
        equipos_ordenados,
        key="equipo_local_select",
    )

    equipo_visitante = st.selectbox(
        "Equipo visitante",
        equipos_ordenados,
        key="equipo_visitante_select",
    )

    st.info(
        f"Partido seleccionado: **{equipo_local} vs {equipo_visitante}**"
    )
    # -------------------------------------------------
    # Analizar partido
    # -------------------------------------------------
    if st.button("🔍 Analizar partido"):
        indice = st.session_state["indice"]

        partido_df = datos.df[
            (datos.df["equipo_local"] == equipo_local)
            & (datos.df["equipo_visitante"] == equipo_visitante)
        ]

        if partido_df.empty:
            st.error(
                "No existe ese partido con ese orden de local/visitante en la temporada seleccionada."
            )
        else:
            partido = partido_df.iloc[0]

            contexto = construir_contexto_partido(
                indice=indice,
                fecha_partido=partido["fecha"],
                equipo_local=equipo_local,
                equipo_visitante=equipo_visitante,
                liga=partido["liga"],
                temporada=partido["temporada"],
            )

            st.success("✅ Contexto construido correctamente")
            promedio_ref = promedio_referencia(
            contexto.historico_local.partidos,
            contexto.historico_visitante.partidos,
            )

            fuerza_local = calcular_fuerza(
            contexto.historico_local.partidos,
            promedio_ref,
            )

            fuerza_visitante = calcular_fuerza(
            contexto.historico_visitante.partidos,
            promedio_ref,
            )

            goles_esperados = calcular_goles_esperados(
            fuerza_local,
            fuerza_visitante,
            promedio_ref,
            )
            probabilidades = calcular_probabilidades(goles_esperados)

            matriz_marcadores = calcular_matriz_marcadores(
                goles_esperados
            )

            probabilidades_1x2 = calcular_1x2(
                matriz_marcadores
            )

            top_marcadores = sorted(
                matriz_marcadores.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5]
            # -----------------------------
            # Cuotas actuales (The Odds API)
            # -----------------------------

            cuotas_partido = None

            try:
                datos_api = obtener_cuotas_epl()

                partidos_api = normalizar_cuotas_epl(
                    datos_api
                )

                cuotas_partido = buscar_partido(
                    partidos_api,
                    equipo_local,
                    equipo_visitante,
                )

            except Exception as e:
                st.warning(
                    f"No fue posible consultar las cuotas: {e}"
                )
                
            st.session_state["analisis"] = {
                "contexto": contexto,
                "goles_esperados": goles_esperados,
                "probabilidades": probabilidades,
                "probabilidades_1x2": probabilidades_1x2,
                "top_marcadores": top_marcadores,
                "fuerza_local": fuerza_local,
                "fuerza_visitante": fuerza_visitante,
                "equipo_local": equipo_local,
                "equipo_visitante": equipo_visitante,
                "cuotas_partido": cuotas_partido,
            
            }
 # ==========================================
    # Mostrar análisis guardado
    # ==========================================

    if "analisis" in st.session_state:

        analisis = st.session_state["analisis"]

        contexto = analisis["contexto"]
        goles_esperados = analisis["goles_esperados"]
        probabilidades = analisis["probabilidades"]
        probabilidades_1x2 = analisis["probabilidades_1x2"]
        top_marcadores = analisis["top_marcadores"]
        fuerza_local = analisis["fuerza_local"]
        fuerza_visitante = analisis["fuerza_visitante"]
        equipo_local = analisis["equipo_local"]
        equipo_visitante = analisis["equipo_visitante"]
        cuotas_partido = analisis["cuotas_partido"]

        # ==========================================
        # Goles esperados
        # ==========================================

        st.divider()
        st.subheader("⚽ Goles esperados")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                f"🏠 {equipo_local}",
                f"{goles_esperados.lambda_local:.2f}",
                help="Goles esperados del equipo local.",
            )

        with col2:
            st.metric(
                f"🚗 {equipo_visitante}",
                f"{goles_esperados.lambda_visitante:.2f}",
                help="Goles esperados del equipo visitante.",
            )

        # ==========================================
        # Resultado 1X2
        # ==========================================

        st.divider()
        st.subheader("🏆 Resultado 1X2")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                f"🏠 {equipo_local}",
                f"{probabilidades_1x2['local'] * 100:.1f}%",
            )

        with col2:
            st.metric(
                "🤝 Empate",
                f"{probabilidades_1x2['empate'] * 100:.1f}%",
            )

        with col3:
            st.metric(
                f"🚗 {equipo_visitante}",
                f"{probabilidades_1x2['visitante'] * 100:.1f}%",
            )

        # ==========================================
        # Mercados
        # ==========================================

        st.divider()
        st.subheader("🎲 Mercados")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Over 1.5", f"{probabilidades.over_15 * 100:.1f}%")

        with col2:
            st.metric("Over 2.5", f"{probabilidades.over_25 * 100:.1f}%")

        with col3:
            st.metric("Under 2.5", f"{probabilidades.under_25 * 100:.1f}%")

        with col4:
            st.metric("BTTS Sí", f"{probabilidades.btts_si * 100:.1f}%")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Under 1.5", f"{probabilidades.under_15 * 100:.1f}%")

        with col2:
            st.metric("Over 3.5", f"{probabilidades.over_35 * 100:.1f}%")

        with col3:
            st.metric("Under 3.5", f"{probabilidades.under_35 * 100:.1f}%")

        with col4:
            st.metric("BTTS No", f"{probabilidades.btts_no * 100:.1f}%")
        

        st.divider()

        st.subheader("🏪 Cuotas API")

        st.write(cuotas_partido)     

        # ==========================================
        # Value Bet
        # ==========================================

        st.divider()
        st.subheader("💰 Value Bet")

        mercados_value = {
            f"{equipo_local} gana": probabilidades_1x2["local"],
            "Empate": probabilidades_1x2["empate"],
            f"{equipo_visitante} gana": probabilidades_1x2["visitante"],
            "Over 2.5": probabilidades.over_25,
            "Under 2.5": probabilidades.under_25,
            "BTTS Sí": probabilidades.btts_si,
        }

        mercado_seleccionado = st.selectbox(
            "Mercado",
            list(mercados_value.keys()),
            key="mercado_value",
        )

        cuota = st.number_input(
            "Cuota decimal",
            min_value=1.01,
            value=1.80,
            step=0.01,
            key="cuota_value",
        )

        prob_modelo = mercados_value[mercado_seleccionado]

        resultado_value = evaluar_value_bet(
            prob_modelo,
            cuota,
        )
        cuota_justa = 1 / prob_modelo if prob_modelo > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Probabilidad modelo",
                f"{resultado_value.prob_sistema_pct:.1f}%",
            )

        with col2:
            st.metric(
                "Probabilidad implícita",
                f"{resultado_value.prob_implicita_pct:.1f}%",
            )

        with col3:
            st.metric(
                "Cuota Justa",
                f"{cuota_justa:.2f}",
            )
        
        with col4:
            st.metric(
                "Edge / Value",
                f"{resultado_value.value_pct:+.1f}%",
            )

        if resultado_value.hay_value:
            st.success("🔥 Hay Value Bet según el modelo.")
        else:
            st.info("No hay Value Bet a esa cuota.")

        # ==========================================
        # Marcadores
        # ==========================================

        st.divider()
        st.subheader("🎯 Marcadores más probables")

        for (goles_local, goles_visitante), probabilidad in top_marcadores:
            st.write(
                f"**{goles_local} - {goles_visitante}** → "
                f"{probabilidad * 100:.1f}%"
            )

        # ==========================================
        # Fuerzas
        # ==========================================

        st.divider()
        st.subheader("💪 Fuerzas del equipo")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"### 🏠 {equipo_local}")
            st.metric("Ataque", f"{fuerza_local.ataque:.2f}")
            st.metric("Defensa", f"{fuerza_local.defensa:.2f}")
            st.metric(
                "Ataque relativo",
                f"{fuerza_local.ataque_relativo:.2f}",
            )
            st.metric(
                "Defensa relativa",
                f"{fuerza_local.defensa_relativa:.2f}",
            )

        with col2:
            st.markdown(f"### 🚗 {equipo_visitante}")
            st.metric("Ataque", f"{fuerza_visitante.ataque:.2f}")
            st.metric("Defensa", f"{fuerza_visitante.defensa:.2f}")
            st.metric(
                "Ataque relativo",
                f"{fuerza_visitante.ataque_relativo:.2f}",
            )
            st.metric(
                "Defensa relativa",
                f"{fuerza_visitante.defensa_relativa:.2f}",
            )

        # ==========================================
        # Histórico
        # ==========================================

        st.divider()
        st.subheader(f"📚 Histórico de {equipo_local}")

        local_col1, local_col2 = st.columns(2)

        local_col1.metric(
            "Partidos",
            contexto.historico_local.n_encontrados,
        )

        local_col2.metric(
            "Confianza",
            contexto.historico_local.confianza_historico,
        )

        import pandas as pd

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "GF": p.goles_a_favor,
                        "GC": p.goles_en_contra,
                    }
                    for p in contexto.historico_local.partidos
                ]
            ),
            use_container_width=True,
        )

        st.divider()
        st.subheader(f"📚 Histórico de {equipo_visitante}")

        visit_col1, visit_col2 = st.columns(2)

        visit_col1.metric(
            "Partidos",
            contexto.historico_visitante.n_encontrados,
        )

        visit_col2.metric(
            "Confianza",
            contexto.historico_visitante.confianza_historico,
        )

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "GF": p.goles_a_favor,
                        "GC": p.goles_en_contra,
                    }
                    for p in contexto.historico_visitante.partidos
                ]
            ),
            use_container_width=True,
        )

        if contexto.advertencias:
            st.warning("Advertencias del histórico")

            for advertencia in contexto.advertencias:
                st.write(f"- {advertencia}")            
            