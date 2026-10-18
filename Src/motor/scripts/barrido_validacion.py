"""
Barrido exhaustivo para responder:
1. Cuál es el mejor valor de alpha (ponderación de forma reciente).
2. Cuántos partidos históricos conviene usar.
3. Si la corrección Dixon-Coles mejora el modelo cuando la correlación de
   marcadores bajos existe realmente en los datos.
4. Límites del modelo.
"""
import sys, random
from pathlib import Path
RUTA_SRC = Path(__file__).resolve().parents[2]
if str(RUTA_SRC) not in sys.path:
    sys.path.insert(0, str(RUTA_SRC))

from motor.core.fuerzas import calcular_fuerza, promedio_referencia
from motor.core.expected_goals import calcular_goles_esperados
from motor.core.poisson_model import calcular_probabilidades
from motor.core.dixon_coles import calcular_probabilidades_dc
from motor.core.metrics import brier_score
from motor.tests.simulador import (generar_universo_equipos, simular_partido,
                                   simular_partido_correlacionado, generar_historico)

PROMEDIO_LIGA = 1.4
N_EQUIPOS = 30
N_PRUEBA = 600
SIGMA = 0.45  # dispersión realista de liga (confirmado en la validación anterior)


def correr_combo(alpha, n_hist, seed_universo=7, seed_pruebas=2026, n_prueba=N_PRUEBA):
    rnd = random.Random(seed_pruebas)
    equipos = generar_universo_equipos(N_EQUIPOS, seed=seed_universo, sigma=SIGMA)
    mercados = ["over_05", "over_15", "over_25", "over_35", "btts_si"]
    preds = {m: [] for m in mercados}
    reales = {m: [] for m in mercados}

    for _ in range(n_prueba):
        local, visitante = rnd.sample(equipos, 2)
        hl = generar_historico(local, equipos, PROMEDIO_LIGA, n_hist, rnd, True)
        hv = generar_historico(visitante, equipos, PROMEDIO_LIGA, n_hist, rnd, False)
        pr = promedio_referencia(hl, hv)
        fl = calcular_fuerza(hl, pr, alpha)
        fv = calcular_fuerza(hv, pr, alpha)
        ge = calcular_goles_esperados(fl, fv, pr)
        probs = calcular_probabilidades(ge)

        gl, gv, _, _ = simular_partido(local, visitante, PROMEDIO_LIGA, rnd)
        total = gl + gv

        preds["over_05"].append(probs.over_05); reales["over_05"].append(1 if total > 0.5 else 0)
        preds["over_15"].append(probs.over_15); reales["over_15"].append(1 if total > 1.5 else 0)
        preds["over_25"].append(probs.over_25); reales["over_25"].append(1 if total > 2.5 else 0)
        preds["over_35"].append(probs.over_35); reales["over_35"].append(1 if total > 3.5 else 0)
        preds["btts_si"].append(probs.btts_si); reales["btts_si"].append(1 if (gl > 0 and gv > 0) else 0)

    brier_promedio = sum(brier_score(preds[m], reales[m]) for m in mercados) / len(mercados)
    return brier_promedio, {m: brier_score(preds[m], reales[m]) for m in mercados}


def barrido_alpha_historico():
    print("=" * 70)
    print("PARTE 1: Barrido de alpha x tamaño de histórico")
    print("(Brier Score promedio de los 5 mercados — menor es mejor)")
    print("=" * 70)
    alphas = [0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
    n_hists = [5, 10, 15, 20, 25, 30]

    resultados = []
    print(f"\n{'alpha':<8}", end="")
    for n in n_hists:
        print(f"n={n:<9}", end="")
    print()

    mejor = (None, None, float("inf"))
    for alpha in alphas:
        print(f"{alpha:<8}", end="")
        for n in n_hists:
            bs, _ = correr_combo(alpha, n)
            print(f"{bs:<11.4f}", end="")
            resultados.append((alpha, n, bs))
            if bs < mejor[2]:
                mejor = (alpha, n, bs)
        print()

    print(f"\nMejor combinación: alpha={mejor[0]}, historico={mejor[1]} partidos, "
          f"Brier promedio={mejor[2]:.4f}")
    return resultados, mejor


def comparar_dixon_coles():
    print("\n" + "=" * 70)
    print("PARTE 2: Poisson puro vs. Dixon-Coles")
    print("(datos simulados CON correlación real en marcadores bajos, rho_real=-0.13)")
    print("=" * 70)

    alpha, n_hist = 0.90, 20
    rho_modelo = -0.10  # valor típico reportado en la literatura (Dixon & Coles, 1997)

    rnd = random.Random(2026)
    equipos = generar_universo_equipos(N_EQUIPOS, seed=7, sigma=SIGMA)
    mercados = ["over_05", "over_15", "over_25", "over_35", "btts_si"]
    preds_puro = {m: [] for m in mercados}
    preds_dc = {m: [] for m in mercados}
    reales = {m: [] for m in mercados}

    for _ in range(N_PRUEBA):
        local, visitante = rnd.sample(equipos, 2)
        hl = generar_historico(local, equipos, PROMEDIO_LIGA, n_hist, rnd, True)
        hv = generar_historico(visitante, equipos, PROMEDIO_LIGA, n_hist, rnd, False)
        pr = promedio_referencia(hl, hv)
        fl = calcular_fuerza(hl, pr, alpha)
        fv = calcular_fuerza(hv, pr, alpha)
        ge = calcular_goles_esperados(fl, fv, pr)

        probs_puro = calcular_probabilidades(ge)
        probs_dc = calcular_probabilidades_dc(ge.lambda_local, ge.lambda_visitante, rho_modelo)

        # Resultado real generado CON correlación verdadera
        gl, gv, _, _ = simular_partido_correlacionado(local, visitante, PROMEDIO_LIGA, rnd,
                                                        rho_verdadero=-0.13)
        total = gl + gv

        preds_puro["over_05"].append(probs_puro.over_05)
        preds_puro["over_15"].append(probs_puro.over_15)
        preds_puro["over_25"].append(probs_puro.over_25)
        preds_puro["over_35"].append(probs_puro.over_35)
        preds_puro["btts_si"].append(probs_puro.btts_si)

        preds_dc["over_05"].append(probs_dc.over_05)
        preds_dc["over_15"].append(probs_dc.over_15)
        preds_dc["over_25"].append(probs_dc.over_25)
        preds_dc["over_35"].append(probs_dc.over_35)
        preds_dc["btts_si"].append(probs_dc.btts_si)

        reales["over_05"].append(1 if total > 0.5 else 0)
        reales["over_15"].append(1 if total > 1.5 else 0)
        reales["over_25"].append(1 if total > 2.5 else 0)
        reales["over_35"].append(1 if total > 3.5 else 0)
        reales["btts_si"].append(1 if (gl > 0 and gv > 0) else 0)

    print(f"\n{'Mercado':<12}{'Brier Poisson puro':>20}{'Brier Dixon-Coles':>20}{'Mejora':>10}")
    print("-" * 62)
    for m in mercados:
        bp = brier_score(preds_puro[m], reales[m])
        bd = brier_score(preds_dc[m], reales[m])
        mejora = (bp - bd) / bp * 100
        print(f"{m:<12}{bp:>20.4f}{bd:>20.4f}{mejora:>9.1f}%")


if __name__ == "__main__":
    barrido_alpha_historico()
    comparar_dixon_coles()
