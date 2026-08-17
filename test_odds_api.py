from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "Src"))

from motor.adapters.odds_api import (
    obtener_cuotas_epl,
    normalizar_cuotas_epl,
    buscar_partido,
)


datos = obtener_cuotas_epl()

partidos = normalizar_cuotas_epl(datos)

partido = buscar_partido(
    partidos,
    "Arsenal",
    "Coventry City",
)

if partido is None:
    print("Partido no encontrado.")
else:
    print("Partido encontrado:")
    print(
        f"{partido['home_team']} vs "
        f"{partido['away_team']}"
    )

    print()
    print("1X2:")
    print(
        "Local:",
        partido["h2h"][partido["home_team"]],
    )
    print(
        "Empate:",
        partido["h2h"]["Draw"],
    )
    print(
        "Visitante:",
        partido["h2h"][partido["away_team"]],
    )

    print()
    print("Over 2.5:", partido["totals_2_5"]["Over"])
    print("Under 2.5:", partido["totals_2_5"]["Under"])