"""Conector entre fixtures próximos y perfiles calculados desde históricos."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from difflib import get_close_matches
from typing import Mapping

import pandas as pd

from .prediction import PronosticoPartido, predecir_partido
from .team_names import clave_equipo, normalizar_nombre_equipo


@dataclass(frozen=True)
class FixtureProximo:
    """Partido sin resultado, recibido de la UI o de una fuente de fixtures."""

    equipo_local: str
    equipo_visitante: str
    fecha: date | str
    liga: str = ""


def mascara_partidos_anteriores(historico: pd.DataFrame, fecha_limite: object) -> pd.Series:
    """Máscara cronológica segura ante fechas naive, UTC o distintos dtypes pandas.

    La comparación se hace por fecha UTC, acorde con la granularidad de fixture
    que usa la Prediction Card, evitando comparar datetime64 con Timestamp tz-aware.
    """
    if "fecha" not in historico.columns:
        raise ValueError("El histórico no contiene la columna fecha")
    fechas = pd.to_datetime(historico["fecha"], errors="coerce", utc=True).dt.date
    limite = pd.Timestamp(fecha_limite)
    if pd.isna(limite):
        raise ValueError("La fecha límite del partido no es válida")
    if limite.tzinfo is not None:
        limite = limite.tz_convert("UTC")
    return fechas < limite.date()


def normalizar_fixture(fixture: FixtureProximo | Mapping[str, object]) -> FixtureProximo:
    """Acepta el objeto interno o un registro con nombres comunes de columnas."""
    if isinstance(fixture, Mapping):
        def obtener(*nombres: str):
            for nombre in nombres:
                if nombre not in fixture:
                    continue
                valor = fixture[nombre]
                if valor is None or (isinstance(valor, str) and not valor.strip()):
                    continue
                try:
                    if pd.isna(valor):
                        continue
                except (TypeError, ValueError):
                    pass
                return valor
            return None

        local = obtener("equipo_local", "home_team", "HomeTeam", "local")
        visitante = obtener("equipo_visitante", "away_team", "AwayTeam", "visitante")
        fecha = obtener("fecha", "date", "Date", "match_date")
        liga = obtener("liga", "league", "League")
        fixture = FixtureProximo(
            str(local or ""), str(visitante or ""), fecha if fecha is not None else "",
            str(liga) if liga is not None else "",
        )

    local = normalizar_nombre_equipo(fixture.equipo_local)
    visitante = normalizar_nombre_equipo(fixture.equipo_visitante)
    if not local or not visitante or clave_equipo(local) == clave_equipo(visitante):
        raise ValueError("El fixture debe incluir dos equipos distintos y reconocibles.")
    try:
        fecha_timestamp = pd.Timestamp(fixture.fecha)
    except (TypeError, ValueError):
        raise ValueError("El fixture no contiene una fecha válida.") from None
    if pd.isna(fecha_timestamp):
        raise ValueError("El fixture no contiene una fecha válida.")
    fecha = fecha_timestamp.date()
    return FixtureProximo(local, visitante, fecha, fixture.liga)


def pronosticar_fixture(
    fixture: FixtureProximo | Mapping[str, object],
    historico: pd.DataFrame,
    *,
    ventana: int = 20,
    iteraciones: int = 1000,
) -> PronosticoPartido:
    """Enlaza fixture → nombres canónicos → muestras por rol → Poisson/Monte Carlo.

    ``predecir_partido`` solo considera resultados anteriores a la fecha del
    fixture, por lo que este conector no introduce información futura.
    """
    partido = normalizar_fixture(fixture)
    if historico.empty:
        raise ValueError("La liga no tiene partidos históricos válidos para generar perfiles.")
    equipos = {
        clave_equipo(nombre): normalizar_nombre_equipo(nombre)
        for columna in ("equipo_local", "equipo_visitante")
        for nombre in historico[columna].dropna().unique()
    }
    faltantes = [
        nombre for nombre in (partido.equipo_local, partido.equipo_visitante)
        if clave_equipo(nombre) not in equipos
    ]
    if faltantes:
        sugerencias = []
        nombres_disponibles = sorted(set(equipos.values()))
        for nombre in faltantes:
            cercanos = get_close_matches(nombre, nombres_disponibles, n=1, cutoff=0.65)
            if cercanos:
                sugerencias.append(f"{nombre} → {cercanos[0]}")
        detalle_sugerencias = (
            " Coincidencias posibles: " + "; ".join(sugerencias) + "."
            if sugerencias else ""
        )
        raise ValueError(
            "No se encontraron estos equipos en el histórico de la liga: "
            + ", ".join(faltantes)
            + ". Revisa los nombres o carga resultados recientes para esos equipos."
            + detalle_sugerencias
        )

    return predecir_partido(
        historico,
        equipos[clave_equipo(partido.equipo_local)],
        equipos[clave_equipo(partido.equipo_visitante)],
        partido.fecha,
        ventana=ventana,
        iteraciones=iteraciones,
    )
