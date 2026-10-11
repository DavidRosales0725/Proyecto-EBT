"""
Valida el motor estadístico completo con partidos simulados a partir de un
proceso Poisson VERDADERO conocido.

Metodología:
1. Se crea un universo de equipos con fuerzas reales (desconocidas para el motor).
2. Para cada partido de prueba, se genera un histórico de N partidos anteriores
   (5-20, configurable) para el local (solo como local) y el visitante (solo
   como visitante), usando el proceso Poisson verdadero.
3. El motor SOLO ve esos históricos (igual que un usuario real ingresándolos
   a mano) y estima fuerzas, lambda, y probabilidades de mercado.
4. Se simula el resultado REAL de ese partido específico (no visto por el
   motor de antemano) y se evalúa si cada mercado se cumplió.
5. Se agregan las métricas (Brier Score, hit rate) sobre las N pruebas.
"""
import random
import sys
sys.path.insert(0, "/home/claude/motor")

from core.fuerzas import calcular_fuerza, promedio_referencia, ResultadoPartido
from core.expected_goals import calcular_goles_esperados
from core.poisson_model import calcular_probabilidades
from core.metrics import brier_score, log_loss, hit_rate
from tests.simulador import generar_universo_equipos, simular_partido, generar_historico

PROMEDIO_LIGA_VERDADERO = 1.4
N_EQUIPOS = 30
N_PARTIDOS_PRUEBA = 500
N_HISTORICO = 15  # partidos previos por rol (rango 5-20 que pediría el usuario real)
ALPHA = 0.90
SIGMA_DISPERSION = 0.45  # dispersión de calidad típica de una liga real


def evaluar_mercados_reales(goles_local: int, goles_visitante: int) -> dict:
    total = goles_local + goles_visitante
    return {
        "over_05": 1 if total > 0.5 else 0,
        "over_15": 1 if total > 1.5 else 0,
        "over_25": 1 if total > 2.5 else 0,
        "over_35": 1 if total > 3.5 else 0,
        "btts_si": 1 if (goles_local > 0 and goles_visitante > 0) else 0,
    }


def main():
    rnd = random.Random(2026)
    equipos = generar_universo_equipos(N_EQUIPOS, seed=7, sigma=SIGMA_DISPERSION)

    mercados = ["over_05", "over_15", "over_25", "over_35", "btts_si"]
    predicciones = {m: [] for m in mercados}
    reales = {m: [] for m in mercados}

    for _ in range(N_PARTIDOS_PRUEBA):
        local, visitante = rnd.sample(equipos, 2)

        # Históricos que el motor SÍ puede ver (respetando rol local/visitante)
        hist_local = generar_historico(local, equipos, PROMEDIO_LIGA_VERDADERO,
                                        N_HISTORICO, rnd, rol_local=True)
        hist_visitante = generar_historico(visitante, equipos, PROMEDIO_LIGA_VERDADERO,
                                            N_HISTORICO, rnd, rol_local=False)

        # --- Motor: solo usa los históricos, no conoce las fuerzas verdaderas ---
        prom_ref = promedio_referencia(hist_local, hist_visitante)
        fuerza_local = calcular_fuerza(hist_local, prom_ref, ALPHA)
        fuerza_visitante = calcular_fuerza(hist_visitante, prom_ref, ALPHA)
        goles_esp = calcular_goles_esperados(fuerza_local, fuerza_visitante, prom_ref)
        probs = calcular_probabilidades(goles_esp)

        # --- Resultado real del partido de prueba (proceso verdadero) ---
        gl, gv, _, _ = simular_partido(local, visitante, PROMEDIO_LIGA_VERDADERO, rnd)
        reales_partido = evaluar_mercados_reales(gl, gv)

        predicciones["over_05"].append(probs.over_05)
        predicciones["over_15"].append(probs.over_15)
        predicciones["over_25"].append(probs.over_25)
        predicciones["over_35"].append(probs.over_35)
        predicciones["btts_si"].append(probs.btts_si)

        for m in mercados:
            reales[m].append(reales_partido[m])

    print(f"Validación del motor sobre {N_PARTIDOS_PRUEBA} partidos simulados")
    print(f"(históricos de {N_HISTORICO} partidos por rol, alpha={ALPHA})\n")
    print(f"{'Mercado':<12}{'Brier Score':>14}{'Log Loss':>12}{'Hit Rate@50%':>15}{'% Real (base)':>16}")
    print("-" * 69)
    for m in mercados:
        bs = brier_score(predicciones[m], reales[m])
        ll = log_loss(predicciones[m], reales[m])
        hr = hit_rate(predicciones[m], reales[m], umbral=0.5)
        base_real = sum(reales[m]) / len(reales[m]) * 100
        print(f"{m:<12}{bs:>14.4f}{ll:>12.4f}{hr*100:>14.1f}%{base_real:>15.1f}%")

    print("\n--- Comparación contra modelo ingenuo (predice siempre la frecuencia base) ---")
    print(f"{'Mercado':<12}{'Brier Motor':>14}{'Brier Ingenuo':>16}{'Mejora':>10}")
    print("-" * 52)
    for m in mercados:
        bs_motor = brier_score(predicciones[m], reales[m])
        freq_base = sum(reales[m]) / len(reales[m])
        bs_ingenuo = brier_score([freq_base] * len(reales[m]), reales[m])
        mejora = (bs_ingenuo - bs_motor) / bs_ingenuo * 100
        print(f"{m:<12}{bs_motor:>14.4f}{bs_ingenuo:>16.4f}{mejora:>9.1f}%")

    print("\n--- Calibración: probabilidad predicha (bins) vs frecuencia real ---")
    print("(un modelo bien calibrado debe tener 'predicho' ≈ 'real' en cada bin)\n")
    for m in mercados:
        print(f"Mercado: {m}")
        bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]
        for lo, hi in bins:
            idx = [i for i, p in enumerate(predicciones[m]) if lo <= p < hi]
            if not idx:
                continue
            pred_prom = sum(predicciones[m][i] for i in idx) / len(idx) * 100
            real_prom = sum(reales[m][i] for i in idx) / len(idx) * 100
            print(f"  [{lo:.1f}-{hi:.1f}) n={len(idx):<4} predicho={pred_prom:5.1f}%  real={real_prom:5.1f}%")
        print()


if __name__ == "__main__":
    main()
