import os

import requests
from dotenv import load_dotenv
from motor.adapters.football_data import traducir_equipo


load_dotenv()

ODDS_API_URL = (
    "https://api.the-odds-api.com/v4/sports/soccer_epl/odds/"
)


def obtener_cuotas_epl(
    regiones: str = "uk",
    mercados: str = "h2h,totals",
) -> list[dict]:
    """Obtiene las cuotas actuales de Premier League."""

    api_key = os.getenv("THE_ODDS_API_KEY")

    if not api_key:
        raise RuntimeError(
            "No se encontró THE_ODDS_API_KEY en el entorno."
        )

    params = {
        "apiKey": api_key,
        "regions": regiones,
        "markets": mercados,
        "oddsFormat": "decimal",
    }

    respuesta = requests.get(
        ODDS_API_URL,
        params=params,
        timeout=15,
    )

    respuesta.raise_for_status()

    return respuesta.json()


def normalizar_cuotas_epl(
    partidos: list[dict],
) -> list[dict]:
    """
    Convierte la respuesta de The Odds API
    a una estructura sencilla para nuestro motor.
    """

    partidos_normalizados = []

    for partido in partidos:

        home_team = partido["home_team"]
        away_team = partido["away_team"]

        resultado = {
            "id": partido["id"],
            "home_team": home_team,
            "away_team": away_team,
            "commence_time": partido["commence_time"],
            "h2h": {
                home_team: None,
                "Draw": None,
                away_team: None,
            },
            "totals_2_5": {
                "Over": None,
                "Under": None,
            },
        }

        for bookmaker in partido.get("bookmakers", []):

            for mercado in bookmaker.get("markets", []):

                if mercado["key"] == "h2h":

                    for outcome in mercado.get("outcomes", []):

                        nombre = outcome["name"]
                        cuota = outcome["price"]

                        if nombre in resultado["h2h"]:

                            actual = resultado["h2h"][nombre]

                            if actual is None or cuota > actual:
                                resultado["h2h"][nombre] = cuota

                elif mercado["key"] == "totals":

                    for outcome in mercado.get("outcomes", []):

                        if outcome.get("point") != 2.5:
                            continue

                        nombre = outcome["name"]

                        if nombre not in resultado["totals_2_5"]:
                            continue

                        cuota = outcome["price"]
                        actual = resultado["totals_2_5"][nombre]

                        if actual is None or cuota > actual:
                            resultado["totals_2_5"][nombre] = cuota

        partidos_normalizados.append(resultado)

    return partidos_normalizados


def buscar_partido(
    partidos: list[dict],
    equipo_local: str,
    equipo_visitante: str,
) -> dict | None:

    print("\n========== BUSCANDO ==========")
    print("Motor:", equipo_local, "vs", equipo_visitante)
    print("==============================")

    for partido in partidos:

        local_api = traducir_equipo(partido["home_team"])
        visitante_api = traducir_equipo(partido["away_team"])

        print(f"API: {local_api} vs {visitante_api}")

        if (
            local_api == equipo_local
            and visitante_api == equipo_visitante
        ):
            print("✅ ENCONTRADO")
            return partido

    print("❌ NO ENCONTRADO")
    return None