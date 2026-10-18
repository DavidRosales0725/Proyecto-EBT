import os
import hashlib

import requests
from dotenv import load_dotenv
from ..team_names import normalizar_nombre_equipo
try:
    import streamlit as st
except ModuleNotFoundError:
    st = None

load_dotenv()

BASE_URL = "https://api.football-data.org/v4"


def _obtener_partidos_api(_api_key: str, api_key_fingerprint: str):
    """Consulta cacheable; fallos HTTP/JSON se propagan y no se guardan."""
    headers = {
        "X-Auth-Token": _api_key,
    }

    respuesta = requests.get(
        f"{BASE_URL}/competitions/PL/matches",
        headers=headers,
        timeout=20,
    )

    respuesta.raise_for_status()

    return respuesta.json()


_obtener_partidos_cacheado = (
    st.cache_data(ttl=900, show_spinner=False)(_obtener_partidos_api)
    if st is not None else _obtener_partidos_api
)


def obtener_partidos_premier():
    """Obtiene partidos EPL; el caché expira en 15 minutos."""
    api_key = os.getenv("FOOTBALL_DATA_API_KEY")
    if not api_key:
        raise RuntimeError("No se encontró FOOTBALL_DATA_API_KEY.")
    huella = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    return _obtener_partidos_cacheado(api_key, huella)


def obtener_proximos_partidos():
    """
    Devuelve una lista simplificada de los próximos partidos.
    """

    datos = obtener_partidos_premier()

    partidos = []

    for partido in datos["matches"]:

        partidos.append(
            {
                "local": partido["homeTeam"]["name"],
                "visitante": partido["awayTeam"]["name"],
                "fecha": partido["utcDate"],
            }
        )

    return partidos

TRADUCCION_EQUIPOS = {
    "Arsenal FC": "Arsenal",
    "Chelsea FC": "Chelsea",
    "Liverpool FC": "Liverpool",
    "Manchester City FC": "Manchester City",
    "Manchester United FC": "Manchester United",
    "Tottenham Hotspur FC": "Tottenham",
    "Newcastle United FC": "Newcastle",
    "Brighton & Hove Albion FC": "Brighton",
    "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Aston Villa FC": "Aston Villa",
    "Brentford FC": "Brentford",
    "AFC Bournemouth": "Bournemouth",
    "Nottingham Forest FC": "Nottingham Forest",
    "Leeds United FC": "Leeds",
    "Sunderland AFC": "Sunderland",
    "Burnley FC": "Burnley",
    "Hull City AFC": "Hull",
    "Coventry City FC": "Coventry",
    "Ipswich Town FC": "Ipswich",
}

def traducir_equipo(nombre: str) -> str:
    return normalizar_nombre_equipo(nombre)
