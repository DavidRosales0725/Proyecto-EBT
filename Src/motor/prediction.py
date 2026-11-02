"""Pronóstico rápido de un enfrentamiento con forma reciente ajustada por rival."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date
import hashlib

import numpy as np
import pandas as pd

try:
    import streamlit as st
except ModuleNotFoundError:  # el motor también se ejecuta en scripts y pruebas
    st = None

from .core.expected_goals import GolesEsperados
from .core.poisson_model import calcular_1x2, calcular_matriz_marcadores, calcular_probabilidades
from .core.pesos import DEFAULT_ALPHA, promedio_ponderado
from .team_names import normalizar_nombre_equipo

PESO_TEMPORADA_ACTUAL_INICIAL = 0.70
PARTIDOS_PARA_PESO_ACTUAL_COMPLETO = 8
XG_MINIMO = 0.05
XG_MAXIMO = 3.5


def peso_temporada_actual(partidos_actuales: int, *, peso_inicial: float = PESO_TEMPORADA_ACTUAL_INICIAL,
                          partidos_objetivo: int = PARTIDOS_PARA_PESO_ACTUAL_COMPLETO) -> float:
    """Peso de temporada actual: 70% al principio y 100% desde el octavo partido.

    Con cero partidos actuales el peso es cero, así no se descartan los datos previos.
    """
    if partidos_actuales < 0 or partidos_objetivo < 1 or not 0 <= peso_inicial <= 1:
        raise ValueError("Configuración de ponderación de temporadas inválida")
    if partidos_actuales == 0:
        return 0.0
    return min(1.0, peso_inicial + (1.0 - peso_inicial) * min(partidos_actuales / partidos_objetivo, 1.0))


def normalizar_xg(valor: float, promedio_liga: float = 1.35) -> float:
    """Acota xG a valores profesionales plausibles y reemplaza resultados no finitos."""
    referencia = promedio_liga if np.isfinite(promedio_liga) and promedio_liga > 0 else 1.35
    if not np.isfinite(valor):
        valor = referencia
    return float(np.clip(valor, XG_MINIMO, XG_MAXIMO))


def evaluar_confianza(
    partidos_local_actual: int,
    partidos_visitante_actual: int,
    *,
    partidos_local_anterior: int = 0,
    partidos_visitante_anterior: int = 0,
) -> dict[str, str]:
    """Clasifica cobertura por el rol con menos partidos de la temporada actual."""
    conteos = (partidos_local_actual, partidos_visitante_actual,
               partidos_local_anterior, partidos_visitante_anterior)
    if any(conteo < 0 for conteo in conteos):
        raise ValueError("La cobertura no puede ser negativa")
    cobertura_actual = min(partidos_local_actual, partidos_visitante_actual)
    cobertura_anterior = min(partidos_local_anterior, partidos_visitante_anterior)
    if cobertura_actual >= PARTIDOS_PARA_PESO_ACTUAL_COMPLETO:
        nivel, icono = "Alta", "✅"
        detalle = (
            f"Basado en {partidos_local_actual} partidos actuales del local y "
            f"{partidos_visitante_actual} del visitante"
        )
    elif cobertura_actual >= 4:
        nivel, icono = "Moderada", "ℹ️"
        detalle = "Ponderación parcial de temporada actual"
    else:
        nivel, icono = "Baja", "⚠️"
        detalle = (
            "Muestra pequeña combinada con histórico anterior"
            if cobertura_anterior else "Muestra pequeña de la temporada actual"
        )
    return {
        "nivel": nivel,
        "icono": icono,
        "detalle": detalle,
        "partidos_local_actual": str(partidos_local_actual),
        "partidos_visitante_actual": str(partidos_visitante_actual),
        "partidos_local_anterior": str(partidos_local_anterior),
        "partidos_visitante_anterior": str(partidos_visitante_anterior),
    }


def _cachear_prediccion(funcion):
    """Cachea solo resultados correctos; las excepciones no quedan guardadas."""
    if st is None:
        return funcion
    return st.cache_data(ttl=900, show_spinner=False)(funcion)


@dataclass(frozen=True)
class PronosticoPartido:
    equipo_local: str
    equipo_visitante: str
    fecha_partido: str
    lambda_local: float
    lambda_visitante: float
    victoria_local: float
    empate: float
    victoria_visitante: float
    btts_si: float
    btts_no: float
    over_15: float
    under_15: float
    over_25: float
    under_25: float
    over_35: float
    under_35: float
    marcador_simulado: str
    frecuencia_marcador_simulado: float
    iteraciones: int
    partidos_local: int
    partidos_visitante: int
    partidos_local_temporada_actual: int = 0
    partidos_visitante_temporada_actual: int = 0
    partidos_local_temporada_anterior: int = 0
    partidos_visitante_temporada_anterior: int = 0
    temporadas_combinadas: bool = False
    advertencias: tuple[str, ...] = ()


def _ponderar_ajuste_rival(
    muestra: pd.DataFrame,
    ataque_rival: dict[str, float],
    defensa_rival: dict[str, float],
    ataque_columna: str,
    contra_columna: str,
    alpha: float,
    promedio_liga: float,
    temporada_actual: int | None = None,
) -> tuple[float, float]:
    ataque_ajustado = []
    defensa_ajustada = []
    for partido in muestra.itertuples(index=False):
        rival = partido.equipo_visitante if ataque_columna == "goles_local" else partido.equipo_local
        fuerza_defensa_rival = defensa_rival.get(rival, 1.0)
        fuerza_ataque_rival = ataque_rival.get(rival, 1.0)
        ataque_ajustado.append(float(getattr(partido, ataque_columna)) / max(fuerza_defensa_rival, 0.25))
        defensa_ajustada.append(float(getattr(partido, contra_columna)) / max(fuerza_ataque_rival, 0.25))
    if temporada_actual is None or "_temporada" not in muestra.columns:
        return (
            promedio_ponderado(ataque_ajustado, alpha) if ataque_ajustado else promedio_liga,
            promedio_ponderado(defensa_ajustada, alpha) if defensa_ajustada else promedio_liga,
        )

    def media_temporal(valores: list[float]) -> float:
        temporadas = muestra["_temporada"].astype("Int64")
        actual = temporadas.eq(temporada_actual).fillna(False).to_numpy()
        n_actual = int(actual.sum())
        anteriores = ~actual
        n_anterior = int(anteriores.sum())
        if not n_anterior:
            return promedio_ponderado(valores, alpha)
        if not n_actual:
            return promedio_ponderado(valores, alpha)
        parte_actual = peso_temporada_actual(n_actual)
        serie = np.asarray(valores, dtype=float)
        pesos_recencia = np.power(alpha, np.arange(len(serie) - 1, -1, -1, dtype=float))
        medias = []
        for mascara in (actual, anteriores):
            pesos = pesos_recencia[mascara]
            pesos = pesos / pesos.sum()
            medias.append(float(np.average(serie[mascara], weights=pesos)))
        return parte_actual * medias[0] + (1.0 - parte_actual) * medias[1]

    return (
        media_temporal(ataque_ajustado) if ataque_ajustado else promedio_liga,
        media_temporal(defensa_ajustada) if defensa_ajustada else promedio_liga,
    )


@_cachear_prediccion
def predecir_partido(
    historico: pd.DataFrame,
    equipo_local: str,
    equipo_visitante: str,
    fecha_partido: date | str,
    *,
    ventana: int = 20,
    alpha: float = DEFAULT_ALPHA,
    iteraciones: int = 1000,
) -> PronosticoPartido:
    """Estima mercados Poisson y simula marcadores para una selección.

    La ventana usa hasta ``ventana`` partidos recientes del equipo en su rol
    (local/visitante). Los goles se ajustan por la fuerza global observada de
    ataque y defensa de cada rival antes de la fecha del pronóstico.
    """
    if not 10 <= ventana <= 30:
        raise ValueError("La ventana de recencia debe estar entre 10 y 30 partidos")
    if not np.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("alpha debe estar en el rango (0, 1]")
    if not isinstance(iteraciones, int) or iteraciones < 1:
        raise ValueError("iteraciones debe ser un entero positivo")

    requeridas = {"fecha", "equipo_local", "equipo_visitante", "goles_local", "goles_visitante"}
    faltantes = requeridas.difference(historico.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas históricas: {sorted(faltantes)}")

    corte = pd.Timestamp(fecha_partido)
    if corte.tzinfo is not None:
        corte = corte.tz_localize(None)
    df = historico.copy()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", utc=True).dt.tz_localize(None)
    for columna in ("goles_local", "goles_visitante"):
        df[columna] = pd.to_numeric(df[columna], errors="coerce")
    df = df.dropna(subset=["fecha", "goles_local", "goles_visitante"])
    df = df.loc[df["fecha"] < corte].copy()
    local = normalizar_nombre_equipo(equipo_local)
    visitante = normalizar_nombre_equipo(equipo_visitante)
    if not local or not visitante or local == visitante:
        raise ValueError("Selecciona dos equipos distintos y válidos")
    df["equipo_local"] = df["equipo_local"].map(normalizar_nombre_equipo)
    df["equipo_visitante"] = df["equipo_visitante"].map(normalizar_nombre_equipo)
    df = df.sort_values("fecha", kind="stable")
    if df.empty:
        raise ValueError("No hay partidos históricos anteriores a la fecha seleccionada")

    promedio_liga = float((df["goles_local"].sum() + df["goles_visitante"].sum()) / (2 * len(df)))
    if not np.isfinite(promedio_liga) or promedio_liga <= 0:
        promedio_liga = 1.35

    # Estimación global con prior de cinco partidos de promedio de liga para
    # evitar valores extremos en equipos con pocas observaciones.
    appearances = pd.concat([
        df[["equipo_local", "goles_local", "goles_visitante"]].rename(
            columns={"equipo_local": "equipo", "goles_local": "gf", "goles_visitante": "ga"}
        ),
        df[["equipo_visitante", "goles_visitante", "goles_local"]].rename(
            columns={"equipo_visitante": "equipo", "goles_visitante": "gf", "goles_local": "ga"}
        ),
    ], ignore_index=True)
    ratings = appearances.groupby("equipo", sort=False).agg(
        gf=("gf", "sum"), ga=("ga", "sum"), partidos=("gf", "size")
    )
    prior = 5.0
    ataque_rival = ((ratings["gf"] + prior * promedio_liga) /
                    ((ratings["partidos"] + prior) * promedio_liga)).to_dict()
    defensa_rival = ((ratings["ga"] + prior * promedio_liga) /
                     ((ratings["partidos"] + prior) * promedio_liga)).to_dict()

    temporada_actual = _temporada_partido(pd.Timestamp(fecha_partido))
    if "_temporada" not in df.columns:
        df["_temporada"] = df["fecha"].map(_temporada_partido)
    muestra_local = df.loc[df["equipo_local"].eq(local)].tail(ventana)
    muestra_visitante = df.loc[df["equipo_visitante"].eq(visitante)].tail(ventana)
    if muestra_local.empty:
        raise ValueError(f"No hay histórico reciente de {local} como local en esta liga")
    if muestra_visitante.empty:
        raise ValueError(f"No hay histórico reciente de {visitante} como visitante en esta liga")

    ataque_local, defensa_local = _ponderar_ajuste_rival(
        muestra_local, ataque_rival, defensa_rival,
        "goles_local", "goles_visitante", alpha, promedio_liga,
        temporada_actual,
    )
    ataque_visitante, defensa_visitante = _ponderar_ajuste_rival(
        muestra_visitante, ataque_rival, defensa_rival,
        "goles_visitante", "goles_local", alpha, promedio_liga,
        temporada_actual,
    )
    lambda_local_bruto = ataque_local * defensa_visitante / promedio_liga
    lambda_visitante_bruto = ataque_visitante * defensa_local / promedio_liga
    lamb_local = normalizar_xg(lambda_local_bruto, promedio_liga)
    lamb_visitante = normalizar_xg(lambda_visitante_bruto, promedio_liga)
    goles = GolesEsperados(lamb_local, lamb_visitante)

    mercados = calcular_probabilidades(goles)
    matriz_1x2 = calcular_matriz_marcadores(goles, max_goles=12)
    probs_1x2 = calcular_1x2(matriz_1x2)
    suma_1x2 = sum(probs_1x2.values())
    if suma_1x2 > 0:
        probs_1x2 = {nombre: valor / suma_1x2 for nombre, valor in probs_1x2.items()}

    seed_text = f"{local}|{visitante}|{pd.Timestamp(fecha_partido).date()}|{ventana}|{alpha}"
    seed = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    marcadores_local = rng.poisson(lamb_local, size=iteraciones)
    marcadores_visitante = rng.poisson(lamb_visitante, size=iteraciones)
    marcador, conteo = Counter(zip(marcadores_local.tolist(), marcadores_visitante.tolist())).most_common(1)[0]

    advertencias = []
    if lambda_local_bruto > XG_MAXIMO or lambda_visitante_bruto > XG_MAXIMO:
        advertencias.append(
            f"Un xG bruto superó el límite preventivo de {XG_MAXIMO:.1f}; se acotó para evitar una proyección extrema."
        )
    if len(muestra_local) < ventana:
        advertencias.append(f"{local}: {len(muestra_local)} partidos recientes como local disponibles.")
    if len(muestra_visitante) < ventana:
        advertencias.append(f"{visitante}: {len(muestra_visitante)} partidos recientes como visitante disponibles.")

    temporadas_combinadas = (
        df["_temporada"].nunique(dropna=True) > 1
        and
        (muestra_local["_temporada"].nunique(dropna=True) > 1
         or muestra_visitante["_temporada"].nunique(dropna=True) > 1)
    )
    temporadas_local = muestra_local["_temporada"].astype("Int64")
    temporadas_visitante = muestra_visitante["_temporada"].astype("Int64")
    partidos_local_actual = int(temporadas_local.eq(temporada_actual).fillna(False).sum())
    partidos_visitante_actual = int(temporadas_visitante.eq(temporada_actual).fillna(False).sum())
    partidos_local_anterior = int(temporadas_local.lt(temporada_actual).fillna(False).sum())
    partidos_visitante_anterior = int(temporadas_visitante.lt(temporada_actual).fillna(False).sum())
    return PronosticoPartido(
        equipo_local=local,
        equipo_visitante=visitante,
        fecha_partido=pd.Timestamp(fecha_partido).date().isoformat(),
        lambda_local=float(lamb_local),
        lambda_visitante=float(lamb_visitante),
        victoria_local=float(probs_1x2["local"]),
        empate=float(probs_1x2["empate"]),
        victoria_visitante=float(probs_1x2["visitante"]),
        btts_si=float(mercados.btts_si),
        btts_no=float(mercados.btts_no),
        over_15=float(mercados.over_15),
        under_15=float(mercados.under_15),
        over_25=float(mercados.over_25),
        under_25=float(mercados.under_25),
        over_35=float(mercados.over_35),
        under_35=float(mercados.under_35),
        marcador_simulado=f"{marcador[0]}–{marcador[1]}",
        frecuencia_marcador_simulado=float(conteo / iteraciones),
        iteraciones=iteraciones,
        partidos_local=len(muestra_local),
        partidos_visitante=len(muestra_visitante),
        partidos_local_temporada_actual=partidos_local_actual,
        partidos_visitante_temporada_actual=partidos_visitante_actual,
        partidos_local_temporada_anterior=partidos_local_anterior,
        partidos_visitante_temporada_anterior=partidos_visitante_anterior,
        temporadas_combinadas=temporadas_combinadas,
        advertencias=tuple(advertencias),
    )


def _temporada_partido(valor: object) -> int:
    fecha = pd.Timestamp(valor)
    return fecha.year if fecha.month >= 7 else fecha.year - 1
