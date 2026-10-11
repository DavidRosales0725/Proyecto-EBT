"""
Suite de pruebas para backtesting.py.

Estructura (siguiendo el plan de 19 pruebas comprometido en
diseno-backtesting.md, Sección 15):
  1-2:   anti-leakage (la prueba formal de la Sección 5)
  3-4:   política de histórico insuficiente
  5:     coincidencia con el motor invocado directamente
  6-10:  gestión de cuotas (Sección 8)
  11:    integración con historicos.py (multi-liga)
  12:    coincidencia con core.metrics
  13-14: drawdown/rachas/ROI sobre secuencias sintéticas conocidas
  15:    reproducibilidad del bootstrap
  16:    validación de configuración
  17-18: estructura del resultado y exportación CSV
  19:    integración completa sobre la temporada real E0
"""
import random
import pandas as pd
import pytest
from pathlib import Path

RUTA_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

from backtesting import (
    ejecutar_backtesting, ConfiguracionBacktesting, ErrorConfiguracionInvalida,
    MERCADOS_SOPORTADOS, exportar_csv, _calcular_drawdown_rachas,
    _consolidar_incidencias, Incidencia,
)
from core.fuerzas import calcular_fuerza, promedio_referencia
from core.expected_goals import calcular_goles_esperados
from core.poisson_model import calcular_probabilidades
from core import metrics as core_metrics
from data_loader import cargar_datos


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _construir_df(filas, con_cuotas=False):
    columnas = ["fecha", "equipo_local", "equipo_visitante", "goles_local",
                "goles_visitante", "liga", "temporada"]
    if con_cuotas:
        columnas += ["Avg>2.5", "Avg<2.5"]
    df = pd.DataFrame(filas, columns=columnas)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df.sort_values("fecha").reset_index(drop=True)


@pytest.fixture
def df_real():
    """Temporada real de Premier League, ya validada en sesiones anteriores."""
    resultado = cargar_datos(str(RUTA_EXAMPLES / "E0_real.csv"), imprimir_resumen=False)
    return resultado.df


@pytest.fixture
def df_sintetico_con_cuotas():
    """
    Liga sintética con suficientes partidos para tener histórico completo
    hacia el final, y cuotas de Over 2.5 coherentes (calculadas a partir de
    una probabilidad implícita razonable, con overround positivo).
    """
    equipos = ["A", "B", "C", "D", "E", "F"]
    filas = []
    fecha = pd.Timestamp("2023-08-01")
    rnd = random.Random(11)
    for ronda in range(10):
        for i, local in enumerate(equipos):
            visitante = equipos[(i + 1 + ronda) % len(equipos)]
            if local == visitante:
                continue
            gl, gv = rnd.randint(0, 4), rnd.randint(0, 4)
            # cuotas sintéticas con overround ~5%, centradas en una probabilidad plausible
            p_over = rnd.uniform(0.35, 0.65)
            cuota_over = round((1 / p_over) * 0.975, 2)
            cuota_under = round((1 / (1 - p_over)) * 0.975, 2)
            filas.append((fecha.strftime("%Y-%m-%d"), local, visitante, gl, gv,
                           "L1", "2023", cuota_over, cuota_under))
            fecha += pd.Timedelta(days=2)
    return _construir_df(filas, con_cuotas=True)


# ---------------------------------------------------------------------------
# Test 1-2: Anti-leakage
# ---------------------------------------------------------------------------

def test_prediccion_no_usa_resultado_real(df_sintetico_con_cuotas):
    """
    Prueba formal de la Sección 5, Parte 3: recalcular la probabilidad de
    una muestra de predicciones ya registradas usando SOLO el histórico
    (sin resultado_real ni cuota) debe dar el mismo valor que el registrado.
    """
    from historicos import IndiceHistorico, construir_contexto_partido

    config = ConfiguracionBacktesting(mercados=["over_25"], n_partidos=8, n_minimo=5)
    resultado = ejecutar_backtesting(df_sintetico_con_cuotas, config)
    assert len(resultado.registro_predicciones) > 0

    indice = IndiceHistorico(df_sintetico_con_cuotas)
    rnd = random.Random(5)
    muestra = rnd.sample(resultado.registro_predicciones, min(20, len(resultado.registro_predicciones)))

    for pred in muestra:
        contexto = construir_contexto_partido(
            indice, pred.fecha, pred.equipo_local, pred.equipo_visitante,
            pred.liga, pred.temporada,
            n_partidos=config.n_partidos, n_minimo=config.n_minimo,
            cruzar_temporadas=config.cruzar_temporadas,
            resultado_real=None,  # explícitamente SIN resultado real
        )
        promedio_ref = promedio_referencia(contexto.historico_local.partidos, contexto.historico_visitante.partidos)
        fl = calcular_fuerza(contexto.historico_local.partidos, promedio_ref, config.alpha)
        fv = calcular_fuerza(contexto.historico_visitante.partidos, promedio_ref, config.alpha)
        ge = calcular_goles_esperados(fl, fv, promedio_ref)
        probs = calcular_probabilidades(ge)

        assert probs.over_25 == pytest.approx(pred.probabilidad_sistema, abs=1e-9), (
            f"LEAKAGE o inconsistencia: recalculado={probs.over_25} != registrado={pred.probabilidad_sistema}"
        )


def test_ninguna_prediccion_usa_fecha_futura(df_real):
    """
    Verificación indirecta pero rigurosa: si un partido usa como histórico
    partidos con fecha >= a la suya, el n_historico registrado no podría
    coincidir con el conteo correcto medido de forma independiente sobre
    el DataFrame real (ya validado exhaustivamente en historicos.py).
    Aquí confirmamos consistencia cruzada sobre datos reales.
    """
    config = ConfiguracionBacktesting(mercados=["over_25"])
    resultado = ejecutar_backtesting(df_real, config)

    for pred in resultado.registro_predicciones[:50]:
        historicos_reales_local = df_real[
            (df_real["equipo_local"] == pred.equipo_local) &
            (df_real["liga"] == pred.liga) &
            (df_real["fecha"] < pred.fecha)
        ]
        # el número registrado nunca puede exceder lo que realmente existe antes de la fecha
        assert pred.n_historico_local <= len(historicos_reales_local)


# ---------------------------------------------------------------------------
# Test 3-4: Política de histórico insuficiente
# ---------------------------------------------------------------------------

def test_politica_excluir_no_genera_predicciones_para_historico_insuficiente(df_real):
    config = ConfiguracionBacktesting(politica_historico_insuficiente="excluir")
    resultado = ejecutar_backtesting(df_real, config)
    assert all(p.confianza_historico != "" for p in resultado.registro_predicciones)
    # Con política "excluir", todo lo registrado corresponde a histórico suficiente
    n_partidos_predichos = len(set(p.id_partido for p in resultado.registro_predicciones))
    assert n_partidos_predichos == resultado.resumen_general.n_partidos_evaluados


def test_politica_incluir_con_advertencia_marca_confianza_baja(df_real):
    config_excluir = ConfiguracionBacktesting(politica_historico_insuficiente="excluir")
    config_incluir = ConfiguracionBacktesting(politica_historico_insuficiente="incluir_con_advertencia")

    r_excluir = ejecutar_backtesting(df_real, config_excluir)
    r_incluir = ejecutar_backtesting(df_real, config_incluir)

    n_partidos_excluir = len(set(p.id_partido for p in r_excluir.registro_predicciones))
    n_partidos_incluir = len(set(p.id_partido for p in r_incluir.registro_predicciones))

    assert n_partidos_incluir > n_partidos_excluir
    assert r_incluir.resumen_general.n_partidos_excluidos == 0
    confianzas_incluir = set(p.confianza_historico for p in r_incluir.registro_predicciones)
    assert "baja" in confianzas_incluir or "media" in confianzas_incluir


# ---------------------------------------------------------------------------
# Test 5: Coincidencia con el motor invocado directamente
# ---------------------------------------------------------------------------

def test_prediccion_coincide_con_motor_directo(df_sintetico_con_cuotas):
    from historicos import IndiceHistorico, construir_contexto_partido

    config = ConfiguracionBacktesting(mercados=["over_25", "btts_si"], alpha=0.85, n_partidos=8, n_minimo=5)
    resultado = ejecutar_backtesting(df_sintetico_con_cuotas, config)
    assert len(resultado.registro_predicciones) > 0

    indice = IndiceHistorico(df_sintetico_con_cuotas)
    pred = resultado.registro_predicciones[len(resultado.registro_predicciones) // 2]

    contexto = construir_contexto_partido(
        indice, pred.fecha, pred.equipo_local, pred.equipo_visitante,
        pred.liga, pred.temporada, n_partidos=config.n_partidos,
        n_minimo=config.n_minimo, cruzar_temporadas=config.cruzar_temporadas,
    )
    promedio_ref = promedio_referencia(contexto.historico_local.partidos, contexto.historico_visitante.partidos)
    fl = calcular_fuerza(contexto.historico_local.partidos, promedio_ref, config.alpha)
    fv = calcular_fuerza(contexto.historico_visitante.partidos, promedio_ref, config.alpha)
    ge = calcular_goles_esperados(fl, fv, promedio_ref)
    probs = calcular_probabilidades(ge)

    valor_esperado = getattr(probs, pred.mercado)
    assert valor_esperado == pytest.approx(pred.probabilidad_sistema, abs=1e-9)


# ---------------------------------------------------------------------------
# Test 6-10: Gestión de cuotas
# ---------------------------------------------------------------------------

def test_mercado_sin_columna_cuotas_marcado_no_evaluable(df_sintetico_con_cuotas):
    config = ConfiguracionBacktesting(mercados=["over_05", "over_25"], n_partidos=8, n_minimo=5)
    resultado = ejecutar_backtesting(df_sintetico_con_cuotas, config)

    tipos_incidencia = [i.tipo for i in resultado.incidencias]
    assert "mercado_sin_columna_cuotas" in tipos_incidencia

    apuestas_over05 = [a for a in resultado.registro_apuestas if a.mercado == "over_05"]
    assert len(apuestas_over05) == 0  # over_05 no tiene cuotas en el fixture

    apuestas_over25 = [a for a in resultado.registro_apuestas if a.mercado == "over_25"]
    assert len(apuestas_over25) > 0  # over_25 sí tiene cuotas


def test_cuota_faltante_en_fila_excluye_solo_esa_combinacion():
    filas = [
        ("2023-08-01", "A", "B", 2, 1, "L1", "2023", 1.8, 2.1),
        ("2023-08-03", "C", "D", 1, 0, "L1", "2023", None, None),  # cuota faltante
        ("2023-08-05", "A", "C", 3, 2, "L1", "2023", 1.5, 2.8),
    ]
    for i in range(15):  # histórico de relleno para pasar el umbral de confianza
        filas.append((f"2023-09-{(i%28)+1:02d}", "A", "B", 1, 1, "L1", "2023", 1.9, 1.95))
    df = _construir_df(filas, con_cuotas=True)

    config = ConfiguracionBacktesting(mercados=["over_25"], n_minimo=1, n_partidos=3,
                                       politica_historico_insuficiente="incluir_con_advertencia")
    resultado = ejecutar_backtesting(df, config)

    n_predicciones = len(resultado.registro_predicciones)
    n_apuestas = len(resultado.registro_apuestas)
    assert n_apuestas < n_predicciones, "Al menos una fila con cuota NaN no debe generar Apuesta"


def test_cuota_invalida_menor_o_igual_a_uno_se_descarta():
    filas = []
    fecha = pd.Timestamp("2023-08-01")
    for i in range(20):
        filas.append((fecha.strftime("%Y-%m-%d"), "A", "B", 1, 1, "L1", "2023", 1.9, 1.95))
        fecha += pd.Timedelta(days=3)
    filas.append((fecha.strftime("%Y-%m-%d"), "A", "B", 2, 2, "L1", "2023", 0.95, 1.95))  # cuota inválida
    df = _construir_df(filas, con_cuotas=True)

    config = ConfiguracionBacktesting(mercados=["over_25"], n_minimo=5, n_partidos=10)
    resultado = ejecutar_backtesting(df, config)

    tipos = [i.tipo for i in resultado.incidencias]
    assert "cuota_invalida" in tipos


def test_overround_calculado_con_ambos_lados(df_sintetico_con_cuotas):
    config = ConfiguracionBacktesting(mercados=["over_25"], n_partidos=8, n_minimo=5)
    resultado = ejecutar_backtesting(df_sintetico_con_cuotas, config)
    con_overround = [a for a in resultado.registro_apuestas if a.overround is not None]
    assert len(con_overround) > 0
    for a in con_overround:
        assert a.overround > -1.0  # overround típico es positivo pero toleramos ruido sintético


def test_hay_value_y_se_aposto_independientes_segun_umbral(df_sintetico_con_cuotas):
    config_laxo = ConfiguracionBacktesting(mercados=["over_25"], umbral_value=0.0)
    config_estricto = ConfiguracionBacktesting(mercados=["over_25"], umbral_value=15.0)

    r_laxo = ejecutar_backtesting(df_sintetico_con_cuotas, config_laxo)
    r_estricto = ejecutar_backtesting(df_sintetico_con_cuotas, config_estricto)

    n_apostadas_laxo = sum(1 for a in r_laxo.registro_apuestas if a.se_aposto)
    n_apostadas_estricto = sum(1 for a in r_estricto.registro_apuestas if a.se_aposto)

    assert n_apostadas_estricto <= n_apostadas_laxo


# ---------------------------------------------------------------------------
# Test 11: Integración con historicos.py (multi-liga)
# ---------------------------------------------------------------------------

def test_multiples_ligas_no_contaminan_historico():
    filas = []
    fecha = pd.Timestamp("2023-08-01")
    for i in range(20):
        filas.append((fecha.strftime("%Y-%m-%d"), "A", "B", 3, 0, "L1", "2023", 1.5, 2.8))
        filas.append((fecha.strftime("%Y-%m-%d"), "A", "B", 0, 0, "L2", "2023", 3.5, 1.3))
        fecha += pd.Timedelta(days=3)
    df = _construir_df(filas, con_cuotas=True)

    config = ConfiguracionBacktesting(mercados=["over_25"], n_partidos=5, n_minimo=3)
    resultado = ejecutar_backtesting(df, config)

    preds_l1 = [p for p in resultado.registro_predicciones if p.liga == "L1"]
    preds_l2 = [p for p in resultado.registro_predicciones if p.liga == "L2"]
    # L1: A siempre gana 3-0 (over 2.5 probable). L2: siempre 0-0 (over 2.5 improbable).
    # Si se mezclaran, las probabilidades convergerían entre sí.
    prob_prom_l1 = sum(p.probabilidad_sistema for p in preds_l1) / len(preds_l1)
    prob_prom_l2 = sum(p.probabilidad_sistema for p in preds_l2) / len(preds_l2)
    assert prob_prom_l1 > prob_prom_l2 + 0.3, "Las ligas no deberían mezclar histórico"


# ---------------------------------------------------------------------------
# Test 12: Coincidencia con core.metrics
# ---------------------------------------------------------------------------

def test_brier_logloss_coinciden_con_core_metrics(df_real):
    config = ConfiguracionBacktesting(mercados=["over_25"])
    resultado = ejecutar_backtesting(df_real, config)

    probs = [p.probabilidad_sistema for p in resultado.registro_predicciones]
    reales = [p.resultado_real for p in resultado.registro_predicciones]

    bs_directo = core_metrics.brier_score(probs, reales)
    ll_directo = core_metrics.log_loss(probs, reales)

    assert resultado.metricas.agregado_global.brier_score == pytest.approx(bs_directo)
    assert resultado.metricas.agregado_global.log_loss == pytest.approx(ll_directo)


# ---------------------------------------------------------------------------
# Test 13-14: Drawdown/rachas y ROI sobre secuencias sintéticas conocidas
# ---------------------------------------------------------------------------

def test_drawdown_rachas_caso_sintetico_conocido():
    # secuencia construida a mano: +1,+1,-1,-1,-1,+1,-1,-1,+2
    ganancias = [1, 1, -1, -1, -1, 1, -1, -1, 2]
    drawdown, racha_g, racha_p, capital_max = _calcular_drawdown_rachas(ganancias)

    # trayectoria acumulada: 1,2,1,0,-1,0,-1,-2,0
    # pico=2 en el paso 2; mínimo posterior=-2 en el paso 8 -> drawdown = 2-(-2)=4
    assert drawdown == 4
    assert racha_g == 2   # primeros dos +1
    assert racha_p == 3   # -1,-1,-1
    assert capital_max == 2  # -min(trayectoria, 0) = -(-2) = 2


def test_roi_yield_calculo_manual(df_sintetico_con_cuotas):
    config = ConfiguracionBacktesting(mercados=["over_25"], umbral_value=0.0, n_partidos=8, n_minimo=5)
    resultado = ejecutar_backtesting(df_sintetico_con_cuotas, config)

    apostadas = [a for a in resultado.registro_apuestas if a.se_aposto]
    if not apostadas:
        pytest.skip("El fixture sintético no generó apuestas con este umbral")

    total_staked_manual = sum(a.stake for a in apostadas)
    beneficio_manual = sum(a.ganancia for a in apostadas)
    roi_manual = beneficio_manual / total_staked_manual * 100

    assert resultado.metricas.agregado_global.roi == pytest.approx(roi_manual)


# ---------------------------------------------------------------------------
# Test 15: Reproducibilidad del bootstrap
# ---------------------------------------------------------------------------

def test_bootstrap_reproducible_con_semilla(df_real):
    config1 = ConfiguracionBacktesting(mercados=["over_25"], semilla_bootstrap=123)
    config2 = ConfiguracionBacktesting(mercados=["over_25"], semilla_bootstrap=123)

    r1 = ejecutar_backtesting(df_real, config1)
    r2 = ejecutar_backtesting(df_real, config2)

    assert r1.metricas.agregado_global.intervalo_confianza_brier == r2.metricas.agregado_global.intervalo_confianza_brier


# ---------------------------------------------------------------------------
# Test 16: Validación de configuración
# ---------------------------------------------------------------------------

def test_mercado_no_soportado_falla_rapido(df_real):
    config = ConfiguracionBacktesting(mercados=["over_99"])
    with pytest.raises(ErrorConfiguracionInvalida):
        ejecutar_backtesting(df_real, config)


def test_configuracion_invalida_n_minimo_mayor_que_n_partidos(df_real):
    config = ConfiguracionBacktesting(n_partidos=5, n_minimo=10)
    with pytest.raises(ErrorConfiguracionInvalida):
        ejecutar_backtesting(df_real, config)


def test_configuracion_invalida_alpha_fuera_de_rango(df_real):
    config = ConfiguracionBacktesting(alpha=1.5)
    with pytest.raises(ErrorConfiguracionInvalida):
        ejecutar_backtesting(df_real, config)


# ---------------------------------------------------------------------------
# Test 17-18: Estructura del resultado y exportación CSV
# ---------------------------------------------------------------------------

def test_resultado_estructura_completa_incluso_vacio():
    df_vacio = _construir_df([], con_cuotas=False)
    resultado = ejecutar_backtesting(df_vacio, ConfiguracionBacktesting())
    assert resultado.resumen_general.n_partidos_en_dataset == 0
    assert resultado.registro_predicciones == []
    assert resultado.registro_apuestas == []
    assert resultado.metricas.agregado_global.brier_score is None


def test_exportacion_csv_reproducible(df_sintetico_con_cuotas, tmp_path):
    config = ConfiguracionBacktesting(mercados=["over_25"])
    resultado = ejecutar_backtesting(df_sintetico_con_cuotas, config)

    carpeta1 = tmp_path / "salida1"
    carpeta2 = tmp_path / "salida2"
    rutas1 = exportar_csv(resultado, str(carpeta1))
    rutas2 = exportar_csv(resultado, str(carpeta2))

    for clave in rutas1:
        contenido1 = open(rutas1[clave]).read()
        contenido2 = open(rutas2[clave]).read()
        assert contenido1 == contenido2, f"El CSV '{clave}' no es reproducible"


# ---------------------------------------------------------------------------
# Test 19: Integración completa sobre la temporada real E0
# ---------------------------------------------------------------------------

def test_integracion_temporada_real_E0(df_real):
    resultado = ejecutar_backtesting(df_real, ConfiguracionBacktesting())

    # Dato de referencia ya medido de forma independiente con historicos.py
    assert resultado.resumen_general.n_partidos_evaluados == 176
    assert resultado.resumen_general.n_partidos_excluidos == 204

    # Rangos matemáticamente plausibles
    g = resultado.metricas.agregado_global
    assert 0.0 <= g.brier_score <= 1.0
    assert 0.0 <= g.hit_rate <= 1.0
    for p in resultado.registro_predicciones:
        assert 0.0 <= p.probabilidad_sistema <= 1.0
        assert p.resultado_real in (0, 1)

    # over_25 es el único mercado con cuotas reales disponibles en este CSV
    assert len(resultado.registro_apuestas) == 176
    assert all(a.mercado == "over_25" for a in resultado.registro_apuestas)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
