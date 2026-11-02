from datetime import datetime, timedelta
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Src"))

from motor.data_loader import cargar_datos
from motor.fixtures import FixtureProximo, mascara_partidos_anteriores, pronosticar_fixture
from motor.prediction import evaluar_confianza
from motor.adapters.fixtures_api import (
    COMPETICIONES_GRATUITAS,
    leer_token_api,
    obtener_fixtures_proximos,
    obtener_historicos_temporada,
)


_ESTILOS = """
<style>
  #MainMenu,
  header[data-testid="stHeader"],
  footer,
  [data-testid="stToolbar"],
  [data-testid="stDecoration"],
  [data-testid="stStatusWidget"],
  [data-testid="stDeployButton"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
  }

  :root {
    --ebt-bg: #08070c;
    --ebt-surface: #100e16;
    --ebt-surface-raised: #17131f;
    --ebt-violet: #8b5cf6;
    --ebt-violet-bright: #a78bfa;
    --ebt-violet-soft: rgba(139, 92, 246, .18);
    --ebt-text: #f5f3fa;
    --ebt-muted: #aaa5b8;
  }
  .stApp {
    background: radial-gradient(ellipse at 50% -18%, #191126 0%, #0c0a11 42%, var(--ebt-bg) 78%);
    color: var(--ebt-text);
  }
  .block-container { max-width: 1180px; padding-top: 2.2rem; padding-bottom: 3rem; }
  h1, h2, h3 { letter-spacing: -0.035em; }
  [data-testid="stCaptionContainer"] { color: var(--ebt-muted); }
  [data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(145deg, rgba(20, 17, 28, .98), rgba(12, 10, 16, .98));
    border-color: rgba(167, 139, 250, .24) !important;
    border-radius: 18px !important;
    box-shadow: 0 18px 48px rgba(0, 0, 0, .38), 0 0 24px rgba(139, 92, 246, .055);
  }
  [data-testid="stMetric"] {
    min-height: 98px; padding: 14px 16px; border-radius: 14px;
    background: linear-gradient(145deg, rgba(28, 23, 39, .78), rgba(17, 14, 24, .9));
    border: 1px solid rgba(167, 139, 250, .14);
  }
  [data-testid="stMetricLabel"] { color: #b8b1c8; }
  [data-testid="stMetricValue"] { color: var(--ebt-text); font-weight: 650; }
  [data-testid="stMetricDelta"] svg { color: var(--ebt-violet-bright) !important; }
  [data-testid="stSelectbox"], [data-testid="stDateInput"] { margin-bottom: .25rem; }
  [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
  [data-testid="stDateInput"] input {
    background-color: var(--ebt-surface) !important;
    border-color: rgba(167, 139, 250, .24) !important;
    color: var(--ebt-text) !important;
  }
  [data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within,
  [data-testid="stDateInput"] input:focus {
    border-color: var(--ebt-violet-bright) !important;
    box-shadow: 0 0 0 1px var(--ebt-violet) !important;
  }
  [data-testid="stButton"] button,
  [data-testid="stDownloadButton"] button {
    color: #fff !important;
    border: 1px solid rgba(196, 181, 253, .48) !important;
    border-radius: 10px !important;
    background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
    transition: filter .15s ease, transform .15s ease;
  }
  [data-testid="stButton"] button:hover,
  [data-testid="stDownloadButton"] button:hover { filter: brightness(1.14); transform: translateY(-1px); }
  [data-testid="stSlider"] [role="slider"] {
    background: var(--ebt-violet-bright) !important;
    border-color: #ddd6fe !important;
  }
  [data-testid="stProgress"] div[role="progressbar"] > div {
    background: linear-gradient(90deg, #7c3aed, #c4b5fd) !important;
  }
  [data-testid="stAlert"] {
    background: linear-gradient(120deg, rgba(91, 33, 182, .22), rgba(31, 24, 43, .88)) !important;
    border: 1px solid rgba(167, 139, 250, .38) !important;
    border-left: 4px solid var(--ebt-violet-bright) !important;
    color: var(--ebt-text) !important;
    border-radius: 12px !important;
  }
  [data-testid="stAlert"] p { color: var(--ebt-text) !important; }
  [data-testid="stRadio"] [role="radio"] [data-testid="stMarkdownContainer"] { color: var(--ebt-text); }
  [data-testid="stRadio"] input:checked + div { border-color: var(--ebt-violet-bright) !important; }
  [data-testid="stAlert"], [data-testid="stCaptionContainer"],
  [data-testid="stMarkdownContainer"] { overflow-wrap: anywhere; }
  [data-testid="stSelectbox"], [data-testid="stDateInput"],
  [data-testid="stSlider"] { min-width: 0; width: 100%; }
  [data-testid="stRadio"] [role="radiogroup"] { flex-wrap: wrap; gap: .35rem .8rem; }

  @media (max-width: 900px) {
    .block-container { max-width: 100%; padding: 1.6rem 1.25rem 2.25rem; }
    [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: .75rem; }
    [data-testid="stHorizontalBlock"] > [data-testid="column"] {
      flex: 1 1 calc(50% - .75rem) !important;
      width: auto !important; min-width: 0 !important;
    }
    [data-testid="stMetric"] { min-height: 88px; padding: 12px; }
  }

  @media (max-width: 640px) {
    .block-container { padding: 1rem .8rem 1.5rem; }
    h1 {
      max-width: 100%;
      font-size: clamp(1.3rem, 5.6vw, 1.7rem);
      line-height: 1.18;
      text-wrap: balance;
      overflow-wrap: normal;
      word-break: normal;
    }
    h2 { font-size: clamp(1.35rem, 6vw, 1.75rem); }
    h3 { font-size: 1.1rem; }
    [data-testid="stHorizontalBlock"] { gap: .55rem; }
    [data-testid="stHorizontalBlock"] > [data-testid="column"] {
      flex: 1 1 100% !important;
      width: 100% !important; min-width: 0 !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
      border-radius: 14px !important; padding: .15rem;
    }
    [data-testid="stMetric"] { min-height: 0; padding: 10px 12px; border-radius: 12px; }
    [data-testid="stMetricValue"] { font-size: clamp(1.25rem, 6vw, 1.65rem); }
    [data-testid="stMetricLabel"] { font-size: .82rem; }
    [data-testid="stAlert"] { padding: .7rem .8rem; border-radius: 12px; }
    [data-testid="stAlert"] p { font-size: .9rem; line-height: 1.4; }
    [data-testid="stSelectbox"] label, [data-testid="stDateInput"] label,
    [data-testid="stSlider"] label { font-size: .9rem; }
  }

  @media (max-width: 380px) {
    h1 { font-size: 1.25rem; letter-spacing: -0.045em; }
  }
</style>
"""


LIGAS_SUGERIDAS = [
    "Premier League", "LaLiga", "Serie A", "Bundesliga", "Ligue 1",
    "MLS", "MLS Next Pro", "Saudi Pro League",
]
ALIAS_CARPETA = {
    "premierleague": "Premier League",
    "laliga": "LaLiga",
    "seriea": "Serie A",
    "bundesliga": "Bundesliga",
    "ligue1": "Ligue 1",
    "mls": "MLS",
    "mlsnextpro": "MLS Next Pro",
    "saudiproleague": "Saudi Pro League",
}


@st.cache_data(ttl=600, show_spinner=False)
def _inspeccionar_csv(ruta: str, mtime_ns: int, tamano: int) -> tuple[bool, int, str]:
    """Valida esquema y registros; la firma invalida caché al cambiar el CSV."""
    del mtime_ns, tamano
    try:
        resultado = cargar_datos(ruta, imprimir_resumen=False)
        cantidad = len(resultado.df)
        if cantidad == 0:
            return False, 0, "no contiene partidos válidos"
        return True, cantidad, ""
    except Exception as exc:
        return False, 0, str(exc)


def _clave(texto: str) -> str:
    return re.sub(r"[^a-z0-9]", "", texto.casefold())


_LIGAS_SUGERIDAS_CLAVES = {_clave(nombre) for nombre in LIGAS_SUGERIDAS}


@st.cache_data(ttl=600, show_spinner=False)
def _catalogo_historicos_cacheado(
    firmas: tuple[tuple[str, str, int, int], ...],
    errores_fs: tuple[tuple[str, str], ...],
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    """Catalogación cacheada por firma de cada CSV (ruta, mtime, tamaño)."""
    catalogo: dict[str, list[str]] = {}
    incidencias: dict[str, list[str]] = {}
    for liga, ruta, mtime_ns, tamano in firmas:
        correcto, _partidos, detalle = _inspeccionar_csv(ruta, mtime_ns, tamano)
        if correcto:
            catalogo.setdefault(liga, []).append(ruta)
        else:
            incidencias.setdefault(liga, []).append(
                f"{Path(ruta).name}: {detalle or 'sin partidos válidos'}"
            )
    for liga, detalle in errores_fs:
        incidencias.setdefault(liga, []).append(detalle)
    return (
        {liga: tuple(sorted(rutas)) for liga, rutas in catalogo.items()},
        {liga: tuple(detalles) for liga, detalles in incidencias.items()},
    )


def _catalogo_historicos() -> tuple[dict[str, tuple[Path, ...]], dict[str, tuple[str, ...]]]:
    carpeta_datos = ROOT / "Data" / "raw"
    firmas: list[tuple[str, str, int, int]] = []
    errores_fs: list[tuple[str, str]] = []
    if carpeta_datos.exists():
        for carpeta in carpeta_datos.iterdir():
            if not carpeta.is_dir():
                continue
            nombre = ALIAS_CARPETA.get(_clave(carpeta.name), carpeta.name)
            try:
                rutas = sorted(carpeta.rglob("*.csv"))
            except OSError as exc:
                errores_fs.append((nombre, f"No se pudo leer la carpeta {carpeta.name}: {exc}"))
                continue
            for ruta in rutas:
                try:
                    firma = ruta.stat()
                except OSError as exc:
                    errores_fs.append((nombre, f"{ruta.name}: {exc}"))
                    continue
                firmas.append((nombre, str(ruta.resolve()), firma.st_mtime_ns, firma.st_size))
    catalogo_cacheado, incidencias = _catalogo_historicos_cacheado(
        tuple(firmas), tuple(errores_fs)
    )
    return (
        {liga: tuple(Path(ruta) for ruta in rutas) for liga, rutas in catalogo_cacheado.items()},
        incidencias,
    )


@st.cache_data(ttl=1800, show_spinner=False)
def _cargar_historicos(firmas: tuple[tuple[str, int, int], ...]):
    frames = []
    errores = []
    for ruta, _mtime, _tamano in firmas:
        try:
            datos = cargar_datos(ruta, imprimir_resumen=False)
            if not datos.df.empty:
                frame = datos.df.copy()
                marco = re.search(r"(?<!\d)(\d{2}|\d{4})[-_](?:\d{2}|\d{4})(?!\d)", Path(ruta).stem)
                if marco:
                    anio = int(marco.group(1))
                    temporada = anio + 2000 if anio < 100 else anio
                    frame["_temporada"] = temporada
                else:
                    fechas = pd.to_datetime(frame["fecha"], errors="coerce")
                    frame["_temporada"] = fechas.map(lambda fecha: fecha.year if pd.notna(fecha) and fecha.month >= 7 else fecha.year - 1 if pd.notna(fecha) else pd.NA)
                frames.append(frame)
        except Exception as exc:
            errores.append(f"{Path(ruta).name}: {exc}")
    if not frames:
        return pd.DataFrame(), tuple(errores)
    combinado = pd.concat(frames, ignore_index=True)
    combinado = combinado.drop_duplicates(
        subset=["fecha", "equipo_local", "equipo_visitante"], keep="last"
    ).sort_values("fecha", kind="stable").reset_index(drop=True)
    return combinado, tuple(errores)


@st.cache_data(ttl=1800, show_spinner=False)
def _pronostico_cacheado(
    liga: str,
    local: str,
    visitante: str,
    fecha_partido: str,
    ventana: int,
    firmas: tuple[tuple[str, int, int], ...],
    _historico: pd.DataFrame,
):
    return pronosticar_fixture(
        FixtureProximo(local, visitante, fecha_partido, liga),
        _historico,
        ventana=ventana,
        iteraciones=1000,
    )


def _firmas_liga(archivos: tuple[Path, ...]) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (str(ruta.resolve()), ruta.stat().st_mtime_ns, ruta.stat().st_size)
        for ruta in archivos
    )


def _etiqueta_fixture(fixture: FixtureProximo) -> str:
    fecha = pd.Timestamp(fixture.fecha).strftime("%d/%m · %H:%M UTC")
    return f"{fecha}  |  {fixture.equipo_local} vs {fixture.equipo_visitante}"


def render_prediction_card() -> None:
    st.markdown(_ESTILOS, unsafe_allow_html=True)
    st.title("⚽ EBT · Prediction Card")
    st.caption("Pronóstico rápido · Fixtures próximos o selección manual")

    catalogo, incidencias_csv = _catalogo_historicos()
    ligas = sorted(catalogo, key=lambda liga: (_clave(liga) not in _LIGAS_SUGERIDAS_CLAVES, liga.casefold()))
    hoy_bogota = datetime.now(ZoneInfo("America/Bogota")).date()
    modo = st.radio(
        "Fuente de fixtures", ["Modo Manual", "Modo API Automática"],
        horizontal=True, key="prediction_source_mode",
    )
    fixture_api = None
    api_activa = False
    liga_api = None
    historico_api = pd.DataFrame()
    firmas_api: tuple[tuple[str, int, int], ...] = ()
    errores_api_hist: tuple[str, ...] = ()
    fuente_datos = "CSV local"
    actualizado_en = None

    if modo == "Modo API Automática":
        token = leer_token_api()
        if not token:
            st.warning("No está configurado FOOTBALL_DATA_API_KEY. Continúo en modo manual; añade el token para consultar fixtures automáticos.")
        else:
            opciones_api = COMPETICIONES_GRATUITAS
            if not opciones_api:
                st.warning("No hay competiciones API configuradas; continúa en modo manual.")
            else:
                with st.container(border=True):
                    api_cols = st.columns([1.2, 1, 1])
                    competicion_nombre = api_cols[0].selectbox(
                        "Competición API", list(opciones_api), key="prediction_api_competition"
                    )
                    fecha_desde = api_cols[1].date_input(
                        "Desde", value=hoy_bogota + timedelta(days=1), key="prediction_api_from"
                    )
                    fecha_hasta = api_cols[2].date_input(
                        "Hasta", value=hoy_bogota + timedelta(days=7), key="prediction_api_to"
                    )
                if fecha_hasta < fecha_desde:
                    st.warning("La fecha final debe ser igual o posterior a la inicial. Continúo en modo manual.")
                else:
                    info_competicion = opciones_api[competicion_nombre]
                    liga_api = info_competicion["historico"]
                    try:
                        with st.spinner("Consultando partidos e históricos de temporada…"):
                            temporada_actual = hoy_bogota.year if hoy_bogota.month >= 7 else hoy_bogota.year - 1
                            historico_api = obtener_historicos_temporada(
                                info_competicion["codigo"], temporada=temporada_actual, token=token
                            )
                            actualizado_en = historico_api.attrs.get("actualizado_en")
                            try:
                                historico_anterior = obtener_historicos_temporada(
                                    info_competicion["codigo"], temporada=temporada_actual - 1, token=token
                                )
                                if not historico_anterior.empty:
                                    historico_api = pd.concat([historico_anterior, historico_api], ignore_index=True)
                                    historico_api = historico_api.drop_duplicates(
                                        subset=["fecha", "equipo_local", "equipo_visitante"], keep="last"
                                    ).sort_values("fecha", kind="stable").reset_index(drop=True)
                            except Exception as exc:
                                errores_api_hist = (f"No se pudo cargar la temporada anterior desde API: {exc}",)
                            if historico_api.empty:
                                raise ValueError("La API no devolvió resultados finalizados en la temporada actual ni en la anterior.")
                            ultima_fecha_api = pd.Timestamp(historico_api["fecha"].max()).isoformat()
                            firmas_api = ((f"api:{info_competicion['codigo']}:{ultima_fecha_api}", len(historico_api), 0),)
                            candidatos = obtener_fixtures_proximos(
                                info_competicion["codigo"], fecha_desde, fecha_hasta,
                                token=token,
                            )
                        if not candidatos:
                            st.warning("La API no devolvió partidos programados en ese rango. Continúo en modo manual.")
                        else:
                            opciones_fixture = list(range(len(candidatos)))
                            indice_fixture = st.selectbox(
                                "Partido próximo",
                                opciones_fixture,
                                format_func=lambda indice: _etiqueta_fixture(candidatos[indice]),
                                key="prediction_api_fixture",
                            )
                            fixture_api = candidatos[indice_fixture]
                            fuente_datos = "Football-Data.org · temporadas actual y anterior"
                            # Si la temporada API todavía no ofrece al menos cinco muestras
                            # en cada rol, cambia al histórico local si está disponible.
                            partidos_previos_api = mascara_partidos_anteriores(
                                historico_api, fixture_api.fecha
                            )
                            api_locales = historico_api.loc[
                                partidos_previos_api & historico_api["equipo_local"].eq(fixture_api.equipo_local)
                            ]
                            api_visitantes = historico_api.loc[
                                partidos_previos_api & historico_api["equipo_visitante"].eq(fixture_api.equipo_visitante)
                            ]
                            if (len(api_locales) < 5 or len(api_visitantes) < 5) and liga_api in catalogo:
                                firmas_locales = _firmas_liga(catalogo[liga_api])
                                historico_local, errores_carga_local = _cargar_historicos(firmas_locales)
                                if not historico_local.empty:
                                    historico_api = historico_local
                                    firmas_api = firmas_locales
                                    fuente_datos = "CSV local (fallback por cobertura API)"
                                    errores_api_hist += ("Cobertura API insuficiente; se usan históricos CSV locales.", *errores_carga_local)
                                else:
                                    st.warning("La API tiene pocas muestras y los CSV locales no pudieron cargarse. Continúo con el selector manual.")
                            elif len(api_locales) < 5 or len(api_visitantes) < 5:
                                st.warning("La competición no tiene suficientes partidos recientes en API y no hay CSV local para respaldarla. Continúo con el selector manual.")
                            api_activa = (
                                len(historico_api) > 0
                                and (fuente_datos.startswith("CSV local")
                                     or (len(api_locales) >= 5 and len(api_visitantes) >= 5))
                            )
                    except Exception as exc:
                        st.warning(f"No fue posible cargar el calendario ({exc}). Continúo en modo manual.")

    if not api_activa:
        if not ligas:
            st.warning("No hay ligas con CSV históricos válidos en `Data/raw/`. Añade un CSV válido o configura una API disponible para continuar.")
            if incidencias_csv:
                with st.expander("Archivos que requieren revisión"):
                    for nombre_liga, detalles in sorted(incidencias_csv.items()):
                        st.markdown(f"**{nombre_liga}**")
                        for detalle in detalles:
                            st.caption(detalle)
            return
        st.caption("Modo efectivo: selección manual desde los históricos locales.")
        with st.container(border=True):
            fila1 = st.columns([1.2, 1, 1])
            liga = fila1[0].selectbox("Liga", ligas, index=0, key="prediction_manual_league")
            fecha_partido = fila1[1].date_input(
                "Fecha del partido", value=hoy_bogota + timedelta(days=1), key="prediction_manual_date"
            )
            ventana = fila1[2].slider(
                "Partidos recientes por rol", min_value=10, max_value=30, value=20, step=1,
                help="Se toman los partidos más recientes disponibles como local o visitante.",
            )
            firmas = _firmas_liga(catalogo[liga])
            historico, errores_carga = _cargar_historicos(firmas)
            fuente_datos = "CSV local"
            actualizado_en = max((Path(ruta).stat().st_mtime for ruta, _mtime, _tamano in firmas), default=None)
            if historico.empty:
                st.error("No fue posible preparar un histórico utilizable para esta liga.")
                for error in errores_carga:
                    st.caption(error)
                return
            equipos = sorted(set(historico["equipo_local"]).union(historico["equipo_visitante"]))
            if len(equipos) < 2:
                st.warning(f"Los históricos de {liga} solo contienen {len(equipos)} equipo(s); se necesitan al menos dos.")
                return
            fila2 = st.columns(2)
            local = fila2[0].selectbox("Local", equipos, key=f"prediction_local_{_clave(liga)}")
            visitantes = [equipo for equipo in equipos if equipo != local]
            visitante = fila2[1].selectbox(
                "Visitante", visitantes,
                key=f"prediction_visitante_{_clave(liga)}_{_clave(local)}",
            )
            fixture = FixtureProximo(local, visitante, fecha_partido, liga)
    else:
        liga = liga_api
        historico = historico_api
        errores_carga = errores_api_hist
        firmas = firmas_api
        ventana = st.slider(
            "Partidos recientes por rol", min_value=10, max_value=30, value=20, step=1,
            help="Se toman los partidos más recientes disponibles como local o visitante.",
            key="prediction_api_window",
        )
        fixture = fixture_api
        fecha_partido = pd.Timestamp(fixture.fecha).date()

    if errores_carga:
        with st.expander("Avisos al cargar históricos"):
            for error in errores_carga:
                st.warning(error)
    if errores_api_hist:
        with st.expander("Avisos de cobertura de la API"):
            for error in errores_api_hist:
                st.warning(error)
    if liga in incidencias_csv:
        with st.expander(f"CSV omitidos de {liga}"):
            for detalle in incidencias_csv[liga]:
                st.warning(detalle)

    fecha_ultimo_historico = pd.Timestamp(historico["fecha"].max()).date()
    if (fecha_partido - fecha_ultimo_historico).days > 180:
        st.warning(
            f"El histórico llega hasta {fecha_ultimo_historico:%d/%m/%Y}; "
            "el pronóstico extrapola desde datos antiguos."
        )

    try:
        pronostico = _pronostico_cacheado(
            liga, fixture.equipo_local, fixture.equipo_visitante,
            fecha_partido.isoformat(), ventana, firmas, historico,
        )
    except Exception as exc:
        st.warning(f"No se pudo generar el pronóstico para esta selección: {exc}")
        return

    confianza = evaluar_confianza(
        pronostico.partidos_local_temporada_actual,
        pronostico.partidos_visitante_temporada_actual,
        partidos_local_anterior=pronostico.partidos_local_temporada_anterior,
        partidos_visitante_anterior=pronostico.partidos_visitante_temporada_anterior,
    )
    with st.container(border=True):
        encabezado, fecha_col = st.columns([3, 1])
        encabezado.subheader(f"{pronostico.equipo_local}  vs  {pronostico.equipo_visitante}")
        fecha_col.markdown(f"**{fecha_partido:%d/%m/%Y}**")

        texto_badge = f"{confianza['icono']} Confianza {confianza['nivel']} · {confianza['detalle']}"
        if confianza["nivel"] == "Alta":
            st.success(texto_badge)
        elif confianza["nivel"] == "Moderada":
            st.info(texto_badge)
        else:
            st.warning(texto_badge)
        st.caption(
            f"Temporada actual — Local: {pronostico.partidos_local_temporada_actual}; "
            f"Visitante: {pronostico.partidos_visitante_temporada_actual} partidos. "
            f"Temporada anterior — Local: {pronostico.partidos_local_temporada_anterior}; "
            f"Visitante: {pronostico.partidos_visitante_temporada_anterior} partidos."
        )

        st.markdown("##### Goles esperados · xG")
        xg_local, xg_visitante = st.columns(2)
        xg_local.metric(pronostico.equipo_local, f"{pronostico.lambda_local:.2f}")
        xg_visitante.metric(pronostico.equipo_visitante, f"{pronostico.lambda_visitante:.2f}")

        st.markdown("##### Resultado 1X2")
        c_local, c_empate, c_visitante = st.columns(3)
        c_local.metric("Victoria local", f"{pronostico.victoria_local:.1%}")
        c_empate.metric("Empate", f"{pronostico.empate:.1%}")
        c_visitante.metric("Victoria visitante", f"{pronostico.victoria_visitante:.1%}")

        st.markdown("##### Ambos marcan · BTTS")
        c_si, c_no = st.columns(2)
        c_si.metric("Sí", f"{pronostico.btts_si:.1%}")
        c_no.metric("No", f"{pronostico.btts_no:.1%}")

        st.markdown("##### Rangos de goles")
        o15, o25, o35 = st.columns(3)
        o15.metric("Over 1.5", f"{pronostico.over_15:.1%}", delta=f"Under {pronostico.under_15:.1%}", delta_color="off")
        o25.metric("Over 2.5", f"{pronostico.over_25:.1%}", delta=f"Under {pronostico.under_25:.1%}", delta_color="off")
        o35.metric("Over 3.5", f"{pronostico.over_35:.1%}", delta=f"Under {pronostico.under_35:.1%}", delta_color="off")

        st.markdown(
            "<div style='padding:1rem 1.2rem;border-radius:0.8rem;background:#181122;"
            "border:1px solid #8b5cf6;text-align:center;margin-top:0.5rem;box-shadow:0 0 24px rgba(139,92,246,.12)'>"
            f"<span style='color:#c4b5fd;font-size:0.9rem'>MARCADOR MÁS PROBABLE · {pronostico.iteraciones:,} SIMULACIONES</span><br>"
            f"<span style='font-size:2rem;font-weight:700;color:#faf8ff'>{pronostico.marcador_simulado}</span>"
            f"<span style='color:#c4b5fd'> · {pronostico.frecuencia_marcador_simulado:.1%} de las simulaciones</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    for aviso in pronostico.advertencias:
        st.warning(aviso)
    if isinstance(actualizado_en, (int, float)):
        actualizado_texto = datetime.fromtimestamp(actualizado_en).strftime("%d/%m/%Y %H:%M")
    elif actualizado_en:
        actualizado_texto = pd.Timestamp(actualizado_en).strftime("%d/%m/%Y %H:%M UTC")
    else:
        actualizado_texto = "fecha no disponible"
    st.caption(
        f"{ventana} partidos recientes por rol · {pronostico.partidos_local} muestras de local · "
        f"{pronostico.partidos_visitante} muestras de visitante · "
        f"fuente: {fuente_datos} · actualizado: {actualizado_texto} · "
        f"histórico hasta {fecha_ultimo_historico:%d/%m/%Y}"
    )
