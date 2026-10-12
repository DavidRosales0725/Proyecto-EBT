import os

import requests
from dotenv import load_dotenv
from motor.team_names import normalizar_nombre_equipo, equipos_equivalentes


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

        local_api = normalizar_nombre_equipo(partido.get("home_team", ""))
        visitante_api = normalizar_nombre_equipo(partido.get("away_team", ""))

        print(f"API: {local_api} vs {visitante_api}")

        if (
            equipos_equivalentes(local_api, equipo_local)
            and equipos_equivalentes(visitante_api, equipo_visitante)
        ):
            print("✅ ENCONTRADO")
            return partido

    print("❌ NO ENCONTRADO")
    return None


def consultar_cuotas_partido(
    equipo_local: str,
    equipo_visitante: str,
) -> dict:
    """Consulta cuotas y devuelve un estado explícito si no están disponibles."""
    try:
        datos_api = obtener_cuotas_epl()
        if not datos_api:
            return {
                "estado": "cuota_no_disponible",
                "cuotas": None,
                "mensaje": "La API no devolvió eventos con cuotas.",
            }

        partidos = normalizar_cuotas_epl(datos_api)
        partido = buscar_partido(partidos, equipo_local, equipo_visitante)
        if partido is None:
            return {
                "estado": "cuota_no_disponible",
                "cuotas": None,
                "mensaje": "No hay cuotas disponibles para este partido.",
            }

        hay_precios = any(partido["h2h"].values()) or any(
            partido["totals_2_5"].values()
        )
        if not hay_precios:
            return {
                "estado": "cuota_no_disponible",
                "cuotas": None,
                "mensaje": "El evento está disponible, pero no tiene precios publicados.",
            }

        return {"estado": "disponible", "cuotas": partido, "mensaje": None}
    except (requests.RequestException, ValueError, RuntimeError, KeyError, TypeError) as exc:
        return {
            "estado": "error_api",
            "cuotas": None,
            "mensaje": f"No fue posible consultar las cuotas: {exc}",
        }
