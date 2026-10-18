import sys, random
from pathlib import Path
RUTA_SRC = Path(__file__).resolve().parents[2]
if str(RUTA_SRC) not in sys.path:
    sys.path.insert(0, str(RUTA_SRC))

from motor.core.fuerzas import calcular_fuerza, promedio_referencia
from motor.core.expected_goals import calcular_goles_esperados
from motor.core.poisson_model import calcular_probabilidades
from motor.core.metrics import brier_score
from motor.tests.simulador_dinamico import (generar_universo_dinamico, generar_historico_dinamico,
                                            jugar_partido_actual)

PROMEDIO_LIGA = 1.4
N_EQUIPOS = 30
N_PRUEBA = 600
VOLATILIDAD = 0.05  # qué tanto cambia la fuerza de un equipo partido a partido


def correr_combo(alpha, n_hist, volatilidad, seed_universo=7, seed_pruebas=2026):
    rnd = random.Random(seed_pruebas)
    equipos = generar_universo_dinamico(N_EQUIPOS, seed=seed_universo)
    mercados = ["over_05", "over_15", "over_25", "over_35", "btts_si"]
    preds = {m: [] for m in mercados}
    reales = {m: [] for m in mercados}

    for _ in range(N_PRUEBA):
        local, visitante = rnd.sample(equipos, 2)
        hl = generar_historico_dinamico(local, equipos, PROMEDIO_LIGA, n_hist, rnd, True, volatilidad)
        hv = generar_historico_dinamico(visitante, equipos, PROMEDIO_LIGA, n_hist, rnd, False, volatilidad)
        pr = promedio_referencia(hl, hv)
        fl = calcular_fuerza(hl, pr, alpha)
        fv = calcular_fuerza(hv, pr, alpha)
        ge = calcular_goles_esperados(fl, fv, pr)
        probs = calcular_probabilidades(ge)

        gl, gv = jugar_partido_actual(local, visitante, PROMEDIO_LIGA, rnd)
        total = gl + gv

        preds["over_05"].append(probs.over_05); reales["over_05"].append(1 if total > 0.5 else 0)
        preds["over_15"].append(probs.over_15); reales["over_15"].append(1 if total > 1.5 else 0)
        preds["over_25"].append(probs.over_25); reales["over_25"].append(1 if total > 2.5 else 0)
        preds["over_35"].append(probs.over_35); reales["over_35"].append(1 if total > 3.5 else 0)
        preds["btts_si"].append(probs.btts_si); reales["btts_si"].append(1 if (gl > 0 and gv > 0) else 0)

    return sum(brier_score(preds[m], reales[m]) for m in mercados) / len(mercados)


def main():
    for volatilidad in [0.03, 0.08, 0.15]:
        print(f"\n=== Volatilidad de forma = {volatilidad} "
              f"({'baja: equipos estables' if volatilidad==0.03 else 'media' if volatilidad==0.08 else 'alta: forma muy cambiante'}) ===")
        alphas = [0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
        n_hists = [10, 15, 20]
        print(f"{'alpha':<8}" + "".join(f"n={n:<9}" for n in n_hists))
        mejor = (None, None, float("inf"))
        for alpha in alphas:
            fila = f"{alpha:<8}"
            for n in n_hists:
                bs = correr_combo(alpha, n, volatilidad)
                fila += f"{bs:<11.4f}"
                if bs < mejor[2]:
                    mejor = (alpha, n, bs)
            print(fila)
        print(f"Mejor: alpha={mejor[0]}, n={mejor[1]}, Brier={mejor[2]:.4f}")


if __name__ == "__main__":
    main()
