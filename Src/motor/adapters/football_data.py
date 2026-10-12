import os

import requests
from dotenv import load_dotenv
from motor.team_names import normalizar_nombre_equipo

load_dotenv()

BASE_URL = "https://api.football-data.org/v4"


def obtener_partidos_premier():
    """
    Obtiene todos los partidos de Premier League.
    """

    api_key = os.getenv("FOOTBALL_DATA_API_KEY")

    if not api_key:
        raise RuntimeError(
            "No se encontró FOOTBALL_DATA_API_KEY."
        )

    headers = {
        "X-Auth-Token": api_key,
    }

    respuesta = requests.get(
        f"{BASE_URL}/competitions/PL/matches",
        headers=headers,
        timeout=20,
    )

    respuesta.raise_for_status()

    return respuesta.json()


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
