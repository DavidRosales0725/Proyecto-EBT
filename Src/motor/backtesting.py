"""
backtesting.py

Módulo de evaluación retrospectiva del motor estadístico sobre temporadas
reales completas. Ejecuta el pipeline completo (histórico -> fuerzas -> λ ->
Poisson/Dixon-Coles -> probabilidades) partido por partido, en orden
cronológico estricto, comparando cada predicción contra el resultado real
y (cuando hay cuotas disponibles) contra el mercado de apuestas.

Capa arquitectónica: Evaluación. Depende de historicos.py, core/ (motor
completo) y decision/value_bet.py. No limpia datos (contrato: recibe la salida
de data_loader.py ya procesada) ni decide ventanas temporales (eso ya lo
resuelve historicos.py).

Diseño aprobado en `diseno-backtesting.md`. Este módulo sigue ese diseño
sin cambios de arquitectura; cualquier decisión de implementación que se
apartara de él está señalada explícitamente en los comentarios y fue
reportada al usuario antes de implementarse.
"""
from __future__ import annotations

import os
import random
import dataclasses
from dataclasses import dataclass, field, asdict, replace
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from historicos import (
    IndiceHistorico, construir_contexto_partido, HistoricoEquipo,
    N_HISTORICO_RECOMENDADO, N_HISTORICO_MINIMO, CRUZAR_TEMPORADAS_DEFAULT,
)
from core.fuerzas import calcular_fuerza, promedio_referencia
from core.expected_goals import calcular_goles_esperados, GolesEsperados
from core.poisson_model import calcular_probabilidades
from core.dixon_coles import calcular_probabilidades_dc
from decision.value_bet import evaluar_value_bet, overround
from core import metrics as core_metrics
from core.pesos import DEFAULT_ALPHA


# ---------------------------------------------------------------------------
# Excepciones
# ---------------------------------------------------------------------------

class ErrorConfiguracionInvalida(Exception):
    """Se lanza cuando ConfiguracionBacktesting tiene valores inconsistentes."""
    pass


# ---------------------------------------------------------------------------
# Registro extensible de mercados soportados (Sección 7 del diseño)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DefinicionMercado:
    nombre: str
    evaluador_resultado: Callable[[int, int], bool]
    # casa -> (columna_cuota_favor, columna_cuota_contra); vacío si no hay
    # columnas de cuotas conocidas para este mercado en football-data.co.uk
    casas_cuotas: Dict[str, Tuple[str, str]] = field(default_factory=dict)


def _resultado_over(umbral: float) -> Callable[[int, int], bool]:
    return lambda gl, gv: (gl + gv) > umbral


MERCADOS_SOPORTADOS: Dict[str, DefinicionMercado] = {
    "over_05": DefinicionMercado("over_05", _resultado_over(0.5)),
    "over_15": DefinicionMercado("over_15", _resultado_over(1.5)),
    "over_25": DefinicionMercado("over_25", _resultado_over(2.5), casas_cuotas={
        "Average": ("Avg>2.5", "Avg<2.5"),
        "Bet365": ("B365>2.5", "B365<2.5"),
        "Pinnacle": ("P>2.5", "P<2.5"),
        "Max": ("Max>2.5", "Max<2.5"),
    }),
    "over_35": DefinicionMercado("over_35", _resultado_over(3.5)),
    "btts_si": DefinicionMercado("btts_si", lambda gl, gv: gl > 0 and gv > 0),
}
# NOTA: solo over_25 tiene columnas de cuotas conocidas porque es el único
# mercado de Over/Under que football-data.co.uk publica de forma estándar
# (verificado contra el CSV real de Premier League ya cargado). Los demás
# mercados son evaluables para calibración (Brier/Log Loss) pero no para
# Value Bet -- comportamiento esperado y diseñado en la Sección 8.

MERCADOS_DEFAULT: List[str] = list(MERCADOS_SOPORTADOS.keys())

# Excluidas deliberadamente del registro de cuotas: las columnas de cierre
# (con "C" insertada tras el prefijo de casa, ej. "B365C>2.5"). Es una
# exclusión por diseño de columnas (whitelist positiva), no un filtro de
# texto -- ver Sección 5, Parte 4 del diseño.


def _extraer_probabilidad(probs, nombre_mercado: str) -> float:
    """
    Los objetos ProbabilidadesMercado (Poisson puro) y ProbabilidadesMercadoDC
    (Dixon-Coles) exponen los mismos 5 nombres de atributo que los mercados
    soportados (over_05, over_15, over_25, over_35, btts_si), así que un
    getattr genérico basta como "extractor" -- si en el futuro un mercado
    nuevo no siguiera esta convención, este es el único punto a modificar.
    """
    return getattr(probs, nombre_mercado)


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

@dataclass
class ConfiguracionBacktesting:
    # Decaimiento por partido (vida media ~6.6 partidos con alpha=0.90).
    # Usar evaluación cronológica para recalibrar por competición.
    alpha: float = DEFAULT_ALPHA
    n_partidos: int = N_HISTORICO_RECOMENDADO
    n_minimo: int = N_HISTORICO_MINIMO
    cruzar_temporadas: bool = CRUZAR_TEMPORADAS_DEFAULT
    mercados: List[str] = field(default_factory=lambda: list(MERCADOS_DEFAULT))
    version_modelo: str = "poisson_puro"          # "poisson_puro" | "dixon_coles"
    rho_dixon_coles: float = -0.10
    politica_historico_insuficiente: str = "excluir"  # "excluir" | "incluir_con_advertencia"
    umbral_value: float = 0.0                       # puntos porcentuales
    casa_cuotas: str = "Average"
    stake: float = 1.0
    fecha_inicio_evaluacion: Optional[pd.Timestamp] = None
    fecha_fin_evaluacion: Optional[pd.Timestamp] = None
    semilla_bootstrap: int = 42


def _validar_configuracion(config: ConfiguracionBacktesting) -> None:
    mercados_invalidos = [m for m in config.mercados if m not in MERCADOS_SOPORTADOS]
    if mercados_invalidos:
        raise ErrorConfiguracionInvalida(
            f"Mercado(s) no soportado(s): {mercados_invalidos}. "
            f"Soportados: {list(MERCADOS_SOPORTADOS.keys())}"
        )
    if config.version_modelo not in {"poisson_puro", "dixon_coles"}:
        raise ErrorConfiguracionInvalida(f"version_modelo inválida: '{config.version_modelo}'")
    if config.politica_historico_insuficiente not in {"excluir", "incluir_con_advertencia"}:
        raise ErrorConfiguracionInvalida(
            f"politica_historico_insuficiente inválida: '{config.politica_historico_insuficiente}'"
        )
    if config.n_partidos <= 0 or config.n_minimo <= 0:
        raise ErrorConfiguracionInvalida("n_partidos y n_minimo deben ser positivos")
    if config.n_minimo > config.n_partidos:
        raise ErrorConfiguracionInvalida("n_minimo no puede ser mayor que n_partidos")
    if not np.isfinite(config.alpha) or not (0.0 < config.alpha <= 1.0):
        raise ErrorConfiguracionInvalida("alpha debe estar en el rango (0, 1]")
    if config.stake <= 0:
        raise ErrorConfiguracionInvalida("stake debe ser positivo")


# ---------------------------------------------------------------------------
# Estructuras de salida (Secciones 3 y 9 del diseño)
# ---------------------------------------------------------------------------

@dataclass
class PrediccionMercado:
    id_partido: str
    fecha: pd.Timestamp
    liga: str
    temporada: str
    equipo_local: str
    equipo_visitante: str
    mercado: str
    probabilidad_sistema: float
    resultado_real: int  # 0 o 1
    confianza_historico: str
    n_historico_local: int
    n_historico_visitante: int
    alpha_usado: float
    n_partidos_config: int
    version_modelo: str


@dataclass
class Apuesta(PrediccionMercado):
    cuota: float = 0.0
    casa_cuotas: str = ""
    probabilidad_implicita: float = 0.0
    overround: Optional[float] = None
    value_pct: float = 0.0
    hay_value: bool = False
    se_aposto: bool = False
    stake: float = 0.0
    ganancia: float = 0.0
    roi_acumulado: Optional[float] = None
    yield_acumulado: Optional[float] = None  # numéricamente igual a roi_acumulado


@dataclass
class PartidoExcluido:
    fecha: pd.Timestamp
    equipo_local: str
    equipo_visitante: str
    liga: str
    motivo: str
    detalle: str


@dataclass
class Incidencia:
    tipo: str
    descripcion: str
    n_ocurrencias: int = 1


@dataclass
class ResumenGeneral:
    n_partidos_en_dataset: int
    n_partidos_evaluados: int
    n_partidos_excluidos: int
    n_predicciones_totales: int
    n_apuestas_simuladas: int
    n_value_bets_detectados: int
    periodo_cubierto: Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]
    ligas_incluidas: List[str]
    temporadas_incluidas: List[str]


@dataclass
class MetricasSegmento:
    n: int
    brier_score: Optional[float]
    log_loss: Optional[float]
    hit_rate: Optional[float]
    roi: Optional[float]
    yield_pct: Optional[float]
    beneficio_total: Optional[float]
    drawdown_maximo: Optional[float]
    racha_ganadora_maxima: Optional[int]
    racha_perdedora_maxima: Optional[int]
    valor_esperado_promedio: Optional[float]
    capital_maximo_requerido: Optional[float]
    ratio_retorno_volatilidad: Optional[float]
    intervalo_confianza_brier: Optional[Tuple[float, float]]
    intervalo_confianza_roi: Optional[Tuple[float, float]]


def _metricas_segmento_vacio(n: int = 0) -> MetricasSegmento:
    return MetricasSegmento(
        n=n, brier_score=None, log_loss=None, hit_rate=None,
        roi=None, yield_pct=None, beneficio_total=None,
        drawdown_maximo=None, racha_ganadora_maxima=None, racha_perdedora_maxima=None,
        valor_esperado_promedio=None, capital_maximo_requerido=None,
        ratio_retorno_volatilidad=None,
        intervalo_confianza_brier=None, intervalo_confianza_roi=None,
    )


@dataclass
class MetricasBacktesting:
    agregado_global: MetricasSegmento
    por_mercado: Dict[str, MetricasSegmento]
    por_liga: Dict[str, MetricasSegmento]
    por_temporada: Dict[str, MetricasSegmento]
    por_confianza_historico: Dict[str, MetricasSegmento]
    curva_calibracion: Dict[str, List[Dict]]
    curva_equity: List[Tuple[pd.Timestamp, float]]


@dataclass
class ResultadoBacktesting:
    resumen_general: ResumenGeneral
    registro_predicciones: List[PrediccionMercado]
    registro_apuestas: List[Apuesta]
    partidos_excluidos: List[PartidoExcluido]
    incidencias: List[Incidencia]
    metricas: MetricasBacktesting
    configuracion_usada: ConfiguracionBacktesting


# ---------------------------------------------------------------------------
# Resolución de columnas de cuotas (Sección 8 del diseño)
# ---------------------------------------------------------------------------

def _resolver_cuotas_por_mercado(
    df: pd.DataFrame, config: ConfiguracionBacktesting
) -> Tuple[Dict[str, Optional[Tuple[str, str, str]]], List[Incidencia]]:
    """
    Para cada mercado configurado, determina qué columnas de cuotas usar
    (columna_favor, columna_contra, casa), o None si no hay ninguna
    disponible en este dataset. Se resuelve UNA vez por dataset, no por
    partido -- si un mercado no es evaluable, se reporta una sola Incidencia.
    """
    resolucion: Dict[str, Optional[Tuple[str, str, str]]] = {}
    incidencias: List[Incidencia] = []

    for nombre in config.mercados:
        definicion = MERCADOS_SOPORTADOS[nombre]
        casas = definicion.casas_cuotas

        if not casas:
            resolucion[nombre] = None
            incidencias.append(Incidencia(
                "mercado_sin_columna_cuotas",
                f"El mercado '{nombre}' no tiene columnas de cuotas conocidas "
                f"para ninguna casa (no está en el registro de mercados con odds).",
            ))
            continue

        orden_casas = [config.casa_cuotas] + [c for c in casas if c != config.casa_cuotas]
        encontrado = None
        for casa in orden_casas:
            if casa not in casas:
                continue
            col_favor, col_contra = casas[casa]
            if col_favor in df.columns and col_contra in df.columns:
                encontrado = (col_favor, col_contra, casa)
                break

        if encontrado is None:
            resolucion[nombre] = None
            incidencias.append(Incidencia(
                "mercado_sin_columna_cuotas",
                f"El mercado '{nombre}' no tiene ninguna casa de cuotas disponible "
                f"en este dataset (se buscaron: {list(casas.keys())}).",
            ))
        else:
            resolucion[nombre] = encontrado

    return resolucion, incidencias


# ---------------------------------------------------------------------------
# Evaluación de un partido individual
# ---------------------------------------------------------------------------

_ORDEN_CONFIANZA = {"baja": 0, "media": 1, "alta": 2}


def _confianza_combinada(hist_local: HistoricoEquipo, hist_visitante: HistoricoEquipo) -> str:
    """El eslabón más débil determina la confianza del partido completo."""
    return min(
        [hist_local.confianza_historico, hist_visitante.confianza_historico],
        key=lambda c: _ORDEN_CONFIANZA[c],
    )


def _construir_id_partido(fecha, liga: str, local: str, visitante: str) -> str:
    return f"{liga}_{pd.Timestamp(fecha).date()}_{local}_{visitante}"


def _calcular_probabilidades_partido(goles_esperados: GolesEsperados,
                                      config: ConfiguracionBacktesting):
    if config.version_modelo == "dixon_coles":
        return calcular_probabilidades_dc(
            goles_esperados.lambda_local, goles_esperados.lambda_visitante,
            config.rho_dixon_coles,
        )
    return calcular_probabilidades(goles_esperados)


def _procesar_partido(
    indice: IndiceHistorico, row: pd.Series, config: ConfiguracionBacktesting,
    resolucion_cuotas: Dict[str, Optional[Tuple[str, str, str]]],
) -> Tuple[List[PrediccionMercado], List[Apuesta], Optional[PartidoExcluido], List[Incidencia]]:
    incidencias_locales: List[Incidencia] = []

    contexto = construir_contexto_partido(
        indice, row["fecha"], row["equipo_local"], row["equipo_visitante"],
        row["liga"], row["temporada"],
        n_partidos=config.n_partidos, n_minimo=config.n_minimo,
        cruzar_temporadas=config.cruzar_temporadas,
        resultado_real=(int(row["goles_local"]), int(row["goles_visitante"])),
    )

    suficiente = contexto.historico_local.suficiente and contexto.historico_visitante.suficiente
    if not suficiente and config.politica_historico_insuficiente == "excluir":
        detalle = "; ".join(contexto.advertencias) if contexto.advertencias else "histórico insuficiente"
        excluido = PartidoExcluido(
            fecha=row["fecha"], equipo_local=row["equipo_local"],
            equipo_visitante=row["equipo_visitante"], liga=row["liga"],
            motivo="historico_insuficiente", detalle=detalle,
        )
        return [], [], excluido, incidencias_locales

    confianza = _confianza_combinada(contexto.historico_local, contexto.historico_visitante)

    promedio_ref = promedio_referencia(contexto.historico_local.partidos, contexto.historico_visitante.partidos)
    fuerza_local = calcular_fuerza(contexto.historico_local.partidos, promedio_ref, config.alpha)
    fuerza_visitante = calcular_fuerza(contexto.historico_visitante.partidos, promedio_ref, config.alpha)
    goles_esperados = calcular_goles_esperados(fuerza_local, fuerza_visitante, promedio_ref)
    probs = _calcular_probabilidades_partido(goles_esperados, config)

    id_partido = _construir_id_partido(row["fecha"], row["liga"], row["equipo_local"], row["equipo_visitante"])

    predicciones: List[PrediccionMercado] = []
    apuestas: List[Apuesta] = []
    goles_local, goles_visitante = int(row["goles_local"]), int(row["goles_visitante"])

    for nombre_mercado in config.mercados:
        definicion = MERCADOS_SOPORTADOS[nombre_mercado]
        probabilidad = _extraer_probabilidad(probs, nombre_mercado)
        resultado_bool = definicion.evaluador_resultado(goles_local, goles_visitante)
        resultado_int = 1 if resultado_bool else 0

        prediccion = PrediccionMercado(
            id_partido=id_partido, fecha=row["fecha"], liga=row["liga"], temporada=row["temporada"],
            equipo_local=row["equipo_local"], equipo_visitante=row["equipo_visitante"],
            mercado=nombre_mercado, probabilidad_sistema=probabilidad, resultado_real=resultado_int,
            confianza_historico=confianza,
            n_historico_local=contexto.historico_local.n_encontrados,
            n_historico_visitante=contexto.historico_visitante.n_encontrados,
            alpha_usado=config.alpha, n_partidos_config=config.n_partidos,
            version_modelo=config.version_modelo,
        )
        predicciones.append(prediccion)

        resolucion = resolucion_cuotas.get(nombre_mercado)
        if resolucion is None:
            continue  # incidencia ya registrada a nivel dataset, no se repite

        col_favor, col_contra, casa = resolucion
        valor_favor = row.get(col_favor)
        valor_contra = row.get(col_contra)

        if pd.isna(valor_favor):
            continue  # cuota faltante solo en esta fila -- no genera Apuesta, sí PrediccionMercado

        if valor_favor <= 1.0:
            incidencias_locales.append(Incidencia(
                "cuota_invalida", f"Cuota <= 1.0 para el mercado '{nombre_mercado}' (columna {col_favor})",
            ))
            continue

        cuota = float(valor_favor)
        resultado_value = evaluar_value_bet(probabilidad, cuota)

        overround_val = None
        if valor_contra is not None and not pd.isna(valor_contra) and valor_contra > 1.0:
            overround_val = overround(cuota, float(valor_contra))

        hay_value = resultado_value.hay_value
        se_aposto = resultado_value.value_pct >= config.umbral_value
        ganancia = 0.0
        stake_usado = 0.0
        if se_aposto:
            stake_usado = config.stake
            ganancia = config.stake * (cuota - 1) if resultado_bool else -config.stake

        apuesta = Apuesta(
            **asdict(prediccion),
            cuota=cuota, casa_cuotas=casa,
            probabilidad_implicita=resultado_value.prob_implicita_pct,
            overround=overround_val,
            value_pct=resultado_value.value_pct,
            hay_value=hay_value, se_aposto=se_aposto,
            stake=stake_usado, ganancia=ganancia,
            roi_acumulado=None, yield_acumulado=None,  # se calcula en la agregación
        )
        apuestas.append(apuesta)

    return predicciones, apuestas, None, incidencias_locales


# ---------------------------------------------------------------------------
# Post-procesamiento: acumulados de ROI/Yield e incidencias
# ---------------------------------------------------------------------------

def _calcular_acumulados(apuestas: List[Apuesta]) -> List[Apuesta]:
    """
    Calcula roi_acumulado/yield_acumulado como una pasada posterior sobre la
    secuencia YA ordenada cronológicamente -- decisión de diseño explícita
    (Sección 16): evita acoplar el cálculo al orden de inserción del bucle
    principal.
    """
    total_staked = 0.0
    total_ganancia = 0.0
    resultado: List[Apuesta] = []
    for ap in apuestas:
        if ap.se_aposto:
            total_staked += ap.stake
            total_ganancia += ap.ganancia
        roi = (total_ganancia / total_staked * 100) if total_staked > 0 else None
        resultado.append(replace(ap, roi_acumulado=roi, yield_acumulado=roi))
    return resultado


def _consolidar_incidencias(incidencias: List[Incidencia]) -> List[Incidencia]:
    agregadas: Dict[Tuple[str, str], int] = {}
    for inc in incidencias:
        clave = (inc.tipo, inc.descripcion)
        agregadas[clave] = agregadas.get(clave, 0) + inc.n_ocurrencias
    return [Incidencia(tipo=t, descripcion=d, n_ocurrencias=n) for (t, d), n in agregadas.items()]


# ---------------------------------------------------------------------------
# Drawdown, rachas, capital requerido (sobre una secuencia cronológica de ganancias)
# ---------------------------------------------------------------------------

def _calcular_drawdown_rachas(ganancias: List[float]) -> Tuple[float, int, int, float]:
    acumulado = 0.0
    pico = 0.0
    minimo_acumulado = 0.0
    drawdown_max = 0.0
    racha_actual_g = racha_actual_p = 0
    racha_max_g = racha_max_p = 0

    for g in ganancias:
        acumulado += g
        pico = max(pico, acumulado)
        drawdown_max = max(drawdown_max, pico - acumulado)
        minimo_acumulado = min(minimo_acumulado, acumulado)

        if g > 0:
            racha_actual_g += 1
            racha_actual_p = 0
        elif g < 0:
            racha_actual_p += 1
            racha_actual_g = 0
        else:
            racha_actual_g = 0
            racha_actual_p = 0
        racha_max_g = max(racha_max_g, racha_actual_g)
        racha_max_p = max(racha_max_p, racha_actual_p)

    capital_maximo_requerido = -minimo_acumulado  # minimo_acumulado <= 0 siempre
    return drawdown_max, racha_max_g, racha_max_p, capital_maximo_requerido


# ---------------------------------------------------------------------------
# Bootstrap por partido (no por fila) -- Sección 10 del diseño
# ---------------------------------------------------------------------------

def _bootstrap_ic_brier(df_pred: pd.DataFrame, semilla: int,
                         n_resamples: int = 1000, alpha_ci: float = 0.05
                         ) -> Optional[Tuple[float, float]]:
    # OPTIMIZACIÓN DE RENDIMIENTO (sin cambio de comportamiento matemático):
    # la implementación original hacía pd.concat() de DataFrames de pandas
    # en cada uno de los n_resamples remuestreos -- el overhead de pandas
    # (alineación de índices, reconstrucción de objetos) dominaba el tiempo
    # (~10.5s por segmento medido sobre datos reales). Aquí se precalculan
    # arrays de NumPy una sola vez, y el remuestreo usa np.concatenate +
    # operaciones vectorizadas. La secuencia de llamadas a rnd.randrange()
    # es idéntica a la versión anterior (mismo orden, misma cantidad), por
    # lo que con la misma semilla el resultado es matemáticamente idéntico,
    # no solo "estadísticamente equivalente".
    grupos_np = [
        sub[["probabilidad_sistema", "resultado_real"]].to_numpy(dtype=float)
        for _, sub in df_pred.groupby("id_partido")
    ]
    n_grupos = len(grupos_np)
    if n_grupos < 5:
        return None  # muestra demasiado pequeña para un bootstrap con sentido

    rnd = random.Random(semilla)
    valores = np.empty(n_resamples, dtype=float)
    for k in range(n_resamples):
        indices = [rnd.randrange(n_grupos) for _ in range(n_grupos)]
        muestra = np.concatenate([grupos_np[i] for i in indices], axis=0)
        diferencias = muestra[:, 0] - muestra[:, 1]
        valores[k] = np.mean(diferencias * diferencias)  # == brier_score (mismo cálculo, vectorizado)

    valores.sort()
    lo_idx = int(n_resamples * (alpha_ci / 2))
    hi_idx = min(int(n_resamples * (1 - alpha_ci / 2)), n_resamples - 1)
    return (float(valores[lo_idx]), float(valores[hi_idx]))


def _bootstrap_ic_roi(df_apostadas: pd.DataFrame, semilla: int,
                       n_resamples: int = 1000, alpha_ci: float = 0.05
                       ) -> Optional[Tuple[float, float]]:
    # Misma optimización y misma garantía de reproducibilidad exacta que
    # _bootstrap_ic_brier (ver comentario arriba).
    grupos_np = [
        sub[["stake", "ganancia"]].to_numpy(dtype=float)
        for _, sub in df_apostadas.groupby("id_partido")
    ]
    n_grupos = len(grupos_np)
    if n_grupos < 5:
        return None

    rnd = random.Random(semilla + 1)  # semilla distinta a la de Brier, evita correlación artificial
    valores: List[float] = []
    for _ in range(n_resamples):
        indices = [rnd.randrange(n_grupos) for _ in range(n_grupos)]
        muestra = np.concatenate([grupos_np[i] for i in indices], axis=0)
        staked = muestra[:, 0].sum()
        if staked <= 0:
            continue
        valores.append(float(muestra[:, 1].sum() / staked * 100))
    if not valores:
        return None
    valores.sort()
    lo_idx = int(len(valores) * (alpha_ci / 2))
    hi_idx = min(int(len(valores) * (1 - alpha_ci / 2)), len(valores) - 1)
    return (valores[lo_idx], valores[hi_idx])


# ---------------------------------------------------------------------------
# Agregación de métricas por segmento
# ---------------------------------------------------------------------------

def _dataframe_vacio_apuesta() -> pd.DataFrame:
    campos = [f.name for f in dataclasses.fields(Apuesta)]
    return pd.DataFrame(columns=campos)


def _metricas_de_segmento(df_pred: pd.DataFrame, df_ap: pd.DataFrame,
                           config: ConfiguracionBacktesting) -> MetricasSegmento:
    n = len(df_pred)
    if n == 0:
        return _metricas_segmento_vacio(0)

    probs = df_pred["probabilidad_sistema"].tolist()
    reales = df_pred["resultado_real"].tolist()
    bs = core_metrics.brier_score(probs, reales)
    ll = core_metrics.log_loss(probs, reales)
    hr = core_metrics.hit_rate(probs, reales, umbral=0.5)
    ic_brier = _bootstrap_ic_brier(df_pred, config.semilla_bootstrap)

    roi = yield_pct = beneficio_total = drawdown = None
    racha_g = racha_p = None
    ve_prom = capital_max = ratio_rv = ic_roi = None

    if df_ap is not None and not df_ap.empty and "se_aposto" in df_ap.columns:
        apostadas = df_ap[df_ap["se_aposto"] == True]  # noqa: E712
        if len(apostadas) > 0:
            total_staked = apostadas["stake"].sum()
            beneficio_total = float(apostadas["ganancia"].sum())
            roi = (beneficio_total / total_staked * 100) if total_staked > 0 else None
            yield_pct = roi
            drawdown, racha_g, racha_p, capital_max = _calcular_drawdown_rachas(
                apostadas["ganancia"].tolist()
            )
            ve_prom = float((apostadas["probabilidad_sistema"] * (apostadas["cuota"] - 1) -
                              (1 - apostadas["probabilidad_sistema"])).mean())
            std_ganancia = apostadas["ganancia"].std()
            ratio_rv = (beneficio_total / std_ganancia) if std_ganancia and std_ganancia > 0 else None
            ic_roi = _bootstrap_ic_roi(apostadas, config.semilla_bootstrap)

    return MetricasSegmento(
        n=n, brier_score=bs, log_loss=ll, hit_rate=hr,
        roi=roi, yield_pct=yield_pct, beneficio_total=beneficio_total,
        drawdown_maximo=drawdown, racha_ganadora_maxima=racha_g, racha_perdedora_maxima=racha_p,
        valor_esperado_promedio=ve_prom, capital_maximo_requerido=capital_max,
        ratio_retorno_volatilidad=ratio_rv,
        intervalo_confianza_brier=ic_brier, intervalo_confianza_roi=ic_roi,
    )


def _calcular_curvas_calibracion(df_pred: pd.DataFrame) -> Dict[str, List[Dict]]:
    bins = ((0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01))
    curvas: Dict[str, List[Dict]] = {}
    for mercado, sub in df_pred.groupby("mercado"):
        filas = []
        for lo, hi in bins:
            en_bin = sub[(sub["probabilidad_sistema"] >= lo) & (sub["probabilidad_sistema"] < hi)]
            if len(en_bin) == 0:
                continue
            filas.append({
                "rango": f"[{lo:.1f}-{hi:.1f})",
                "n": len(en_bin),
                "predicho_promedio": float(en_bin["probabilidad_sistema"].mean()),
                "real_frecuencia": float(en_bin["resultado_real"].mean()),
            })
        curvas[mercado] = filas
    return curvas


def _calcular_curva_equity(apuestas: List[Apuesta]) -> List[Tuple[pd.Timestamp, float]]:
    curva = []
    acumulado = 0.0
    for a in apuestas:  # ya están en orden cronológico
        if not a.se_aposto:
            continue
        acumulado += a.ganancia
        curva.append((a.fecha, acumulado))
    return curva


def _calcular_metricas(predicciones: List[PrediccionMercado], apuestas: List[Apuesta],
                        config: ConfiguracionBacktesting) -> MetricasBacktesting:
    if not predicciones:
        vacio = _metricas_segmento_vacio(0)
        return MetricasBacktesting(
            agregado_global=vacio, por_mercado={}, por_liga={}, por_temporada={},
            por_confianza_historico={}, curva_calibracion={}, curva_equity=[],
        )

    df_pred = pd.DataFrame([asdict(p) for p in predicciones])
    df_ap = pd.DataFrame([asdict(a) for a in apuestas]) if apuestas else _dataframe_vacio_apuesta()

    agregado_global = _metricas_de_segmento(df_pred, df_ap, config)

    def _segmento_por(columna: str) -> Dict[str, MetricasSegmento]:
        resultado = {}
        for valor in df_pred[columna].unique():
            sub_pred = df_pred[df_pred[columna] == valor]
            sub_ap = df_ap[df_ap[columna] == valor] if not df_ap.empty else df_ap
            resultado[str(valor)] = _metricas_de_segmento(sub_pred, sub_ap, config)
        return resultado

    por_mercado = _segmento_por("mercado")
    por_liga = _segmento_por("liga")
    por_temporada = _segmento_por("temporada")
    por_confianza_historico = _segmento_por("confianza_historico")

    curva_calibracion = _calcular_curvas_calibracion(df_pred)
    curva_equity = _calcular_curva_equity(apuestas)

    return MetricasBacktesting(
        agregado_global=agregado_global, por_mercado=por_mercado, por_liga=por_liga,
        por_temporada=por_temporada, por_confianza_historico=por_confianza_historico,
        curva_calibracion=curva_calibracion, curva_equity=curva_equity,
    )


# ---------------------------------------------------------------------------
# Resumen general
# ---------------------------------------------------------------------------

def _construir_resumen(df_total: pd.DataFrame, df_eval: pd.DataFrame,
                        predicciones: List[PrediccionMercado], apuestas: List[Apuesta],
                        excluidos: List[PartidoExcluido]) -> ResumenGeneral:
    n_apostadas = sum(1 for a in apuestas if a.se_aposto)
    n_value = sum(1 for a in apuestas if a.hay_value)
    fechas = df_eval["fecha"]
    periodo = (fechas.min(), fechas.max()) if len(fechas) else (None, None)

    return ResumenGeneral(
        n_partidos_en_dataset=len(df_total),
        n_partidos_evaluados=len(df_eval) - len(excluidos),
        n_partidos_excluidos=len(excluidos),
        n_predicciones_totales=len(predicciones),
        n_apuestas_simuladas=n_apostadas,
        n_value_bets_detectados=n_value,
        periodo_cubierto=periodo,
        ligas_incluidas=sorted(df_eval["liga"].unique().tolist()),
        temporadas_incluidas=sorted(df_eval["temporada"].unique().tolist()),
    )


# ---------------------------------------------------------------------------
# Punto de entrada principal
# ---------------------------------------------------------------------------

def ejecutar_backtesting(df_partidos: pd.DataFrame,
                          configuracion: Optional[ConfiguracionBacktesting] = None
                          ) -> ResultadoBacktesting:
    """
    Punto de entrada único del módulo. Recibe el DataFrame ya limpio y
    ordenado de data_loader.py (contrato de entrada, no se re-valida aquí
    más que delegando a historicos.py la verificación de orden cronológico).
    """
    config = configuracion or ConfiguracionBacktesting()
    _validar_configuracion(config)

    resolucion_cuotas, incidencias_dataset = _resolver_cuotas_por_mercado(df_partidos, config)

    # El índice se construye sobre TODO df_partidos, no solo sobre el rango
    # de evaluación -- ver Sección 4, paso 3 del diseño.
    indice = IndiceHistorico(df_partidos)

    df_eval = df_partidos
    if config.fecha_inicio_evaluacion is not None:
        df_eval = df_eval[df_eval["fecha"] >= config.fecha_inicio_evaluacion]
    if config.fecha_fin_evaluacion is not None:
        df_eval = df_eval[df_eval["fecha"] <= config.fecha_fin_evaluacion]
    df_eval = df_eval.sort_values("fecha")

    registro_predicciones: List[PrediccionMercado] = []
    registro_apuestas: List[Apuesta] = []
    partidos_excluidos: List[PartidoExcluido] = []
    incidencias: List[Incidencia] = list(incidencias_dataset)

    for _, row in df_eval.iterrows():
        preds, apts, excluido, inc_locales = _procesar_partido(indice, row, config, resolucion_cuotas)
        registro_predicciones.extend(preds)
        registro_apuestas.extend(apts)
        if excluido is not None:
            partidos_excluidos.append(excluido)
        incidencias.extend(inc_locales)

    incidencias = _consolidar_incidencias(incidencias)
    registro_apuestas = _calcular_acumulados(registro_apuestas)
    metricas = _calcular_metricas(registro_predicciones, registro_apuestas, config)
    resumen = _construir_resumen(df_partidos, df_eval, registro_predicciones, registro_apuestas, partidos_excluidos)

    return ResultadoBacktesting(
        resumen_general=resumen,
        registro_predicciones=registro_predicciones,
        registro_apuestas=registro_apuestas,
        partidos_excluidos=partidos_excluidos,
        incidencias=incidencias,
        metricas=metricas,
        configuracion_usada=config,
    )


# ---------------------------------------------------------------------------
# Exportación e informe (Sección 11 del diseño)
# ---------------------------------------------------------------------------

def exportar_csv(resultado: ResultadoBacktesting, carpeta: str) -> Dict[str, str]:
    os.makedirs(carpeta, exist_ok=True)
    rutas: Dict[str, str] = {}

    df_pred = pd.DataFrame([asdict(p) for p in resultado.registro_predicciones])
    ruta = os.path.join(carpeta, "registro_predicciones.csv")
    df_pred.to_csv(ruta, index=False)
    rutas["predicciones"] = ruta

    df_ap = pd.DataFrame([asdict(a) for a in resultado.registro_apuestas]) if resultado.registro_apuestas \
        else _dataframe_vacio_apuesta()
    ruta = os.path.join(carpeta, "registro_apuestas.csv")
    df_ap.to_csv(ruta, index=False)
    rutas["apuestas"] = ruta

    df_exc = pd.DataFrame([asdict(e) for e in resultado.partidos_excluidos])
    ruta = os.path.join(carpeta, "partidos_excluidos.csv")
    df_exc.to_csv(ruta, index=False)
    rutas["excluidos"] = ruta

    filas_resumen = [{"segmento": "GLOBAL", **asdict(resultado.metricas.agregado_global)}]
    for prefijo, coleccion in [
        ("mercado", resultado.metricas.por_mercado),
        ("liga", resultado.metricas.por_liga),
        ("temporada", resultado.metricas.por_temporada),
        ("confianza", resultado.metricas.por_confianza_historico),
    ]:
        for clave, met in coleccion.items():
            filas_resumen.append({"segmento": f"{prefijo}={clave}", **asdict(met)})
    df_resumen = pd.DataFrame(filas_resumen)
    ruta = os.path.join(carpeta, "resumen_metricas.csv")
    df_resumen.to_csv(ruta, index=False)
    rutas["resumen_metricas"] = ruta

    return rutas


def imprimir_resumen(resultado: ResultadoBacktesting) -> None:
    r = resultado.resumen_general
    g = resultado.metricas.agregado_global
    print("=" * 60)
    print("RESUMEN DE BACKTESTING")
    print("=" * 60)
    print(f"Partidos en dataset:      {r.n_partidos_en_dataset}")
    print(f"Partidos evaluados:       {r.n_partidos_evaluados}")
    print(f"Partidos excluidos:       {r.n_partidos_excluidos}")
    print(f"Predicciones generadas:   {r.n_predicciones_totales}")
    print(f"Apuestas simuladas:       {r.n_apuestas_simuladas}")
    print(f"Value bets detectados:    {r.n_value_bets_detectados}")
    if r.periodo_cubierto[0] is not None:
        print(f"Periodo cubierto:         {r.periodo_cubierto[0].date()} a {r.periodo_cubierto[1].date()}")
    print(f"Ligas incluidas:          {r.ligas_incluidas}")
    print(f"Temporadas incluidas:     {r.temporadas_incluidas}")
    print()
    print("--- Métricas globales ---")
    print(f"Brier Score: {g.brier_score}")
    print(f"Log Loss:    {g.log_loss}")
    print(f"Hit Rate:    {g.hit_rate}")
    if g.intervalo_confianza_brier:
        print(f"IC Brier (95%): {g.intervalo_confianza_brier}")
    print(f"ROI:              {g.roi}")
    print(f"Beneficio total:  {g.beneficio_total}")
    print(f"Drawdown máximo:  {g.drawdown_maximo}")
    if g.intervalo_confianza_roi:
        print(f"IC ROI (95%): {g.intervalo_confianza_roi}")
    print()
    print("--- Incidencias ---")
    for inc in resultado.incidencias:
        print(f"  [{inc.tipo}] {inc.descripcion} (x{inc.n_ocurrencias})")
    print("=" * 60)
