from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "Src"))

from motor.adapters.football_data import obtener_proximos_partidos

partidos = obtener_proximos_partidos()

print()

for partido in partidos:

    print(
        partido["local"],
        "vs",
        partido["visitante"],
    )

    print(partido["fecha"])

    print()