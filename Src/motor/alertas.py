"""Generación de alertas de value bets para la jornada actual.

La primera integración usa calendario y cuotas EPL mediante los adaptadores
existentes. El cálculo de picks es independiente de Streamlit para facilitar
su reutilización desde otras interfaces o scripts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from .adapters.football_data import obtener_partidos_premier
from .adapters.odds_api import obtener_cuotas_epl, normalizar_cuotas_epl
from .core.expected_goals import calcular_goles_esperados
from .core.fuerzas import calcular_fuerza, promedio_referencia
from .core.poisson_model import calcular_1x2, calcular_matriz_marcadores, calcular_probabilidades
from .data_loader import cargar_datos
from .historicos import IndiceHistorico, construir_contexto_partido
from .team_names import equipos_equivalentes, normalizar_nombre_equipo

ZONA_HORARIA_ALERTAS = "America/Bogota"
LIGA_ALERTAS = "Premier League"


@dataclass
class ResultadoBoletinJornada:
    """Boletín ordenado por Edge y avisos de las fuentes consultadas."""
    fecha: str
    apuestas: pd.DataFrame
    avisos: list[str] = field(default_factory=list)


def _obtener_probabilidades(goles_esperados) -> dict[str, float]:
    probs = calcular_probabilidades(goles_esperados)
    p_1x2 = calcular_1x2(calcular_matriz_marcadores(goles_esperados))
    return {
        "Local": p_1x2["local"],
        "Empate": p_1x2["empate"],
        "Visitante": p_1x2["visitante"],
        "Over 2.5": probs.over_25,
        "Under 2.5": probs.under_25,
    }


def _selecciones_cuotas(partido_cuotas: dict, local: str, visitante: str) -> dict[str, Optional[float]]:
    h2h = partido_cuotas.get("h2h", {})
    nombre_local = next((n for n in h2h if n != "Draw" and equipos_equivalentes(n, local)), None)
    nombre_visitante = next((n for n in h2h if n != "Draw" and equipos_equivalentes(n, visitante)), None)
    totales = partido_cuotas.get("totals_2_5", {})
    return {
        "Local": h2h.get(nombre_local) if nombre_local else None,
        "Empate": h2h.get("Draw"),
        "Visitante": h2h.get(nombre_visitante) if nombre_visitante else None,
        "Over 2.5": totales.get("Over"),
        "Under 2.5": totales.get("Under"),
    }


def calcular_alertas_value_bets(
    partidos: list[dict],
    cuotas: list[dict],
    historico: pd.DataFrame,
    fecha_jornada: str,
    *,
    min_edge: float = 0.05,
    alpha: float = 0.90,
    n_partidos: int = 15,
    n_minimo: int = 10,
    bankroll: float = 100.0,
    tipo_stake: str = "kelly_fraccionado",
    fraccion_kelly: float = 0.25,
    stake_fijo: float = 1.0,
    stake_max_pct: float = 0.05,
) -> ResultadoBoletinJornada:
    """Calcula alertas sobre los fixtures y cuotas recibidos.

    ``partidos`` espera elementos con local, visitante y fecha; ``cuotas`` usa
    la estructura de ``normalizar_cuotas_epl``. Edge se expresa en escala 0-1.
    """
    if not 0 <= min_edge <= 0.20:
        raise ValueError("min_edge debe estar entre 0 y 0.20")
    if not 0 < alpha <= 1 or n_partidos < 1 or not 1 <= n_minimo <= n_partidos:
        raise ValueError("Parámetros de histórico o alpha no válidos")
    if bankroll <= 0 or not 0 < stake_max_pct <= 1:
        raise ValueError("Bankroll y límite de stake deben ser positivos")
    if tipo_stake not in {"fijo", "kelly_fraccionado"}:
        raise ValueError("tipo_stake debe ser 'fijo' o 'kelly_fraccionado'")
    if not 0 < fraccion_kelly <= 1 or stake_fijo <= 0:
        raise ValueError("Fracción Kelly y stake fijo deben ser positivos")

    avisos: list[str] = []
    if historico.empty:
        return ResultadoBoletinJornada(fecha_jornada, pd.DataFrame(), ["El histórico de la liga está vacío."])

    hist = historico.copy()
    hist["equipo_local"] = hist["equipo_local"].map(normalizar_nombre_equipo)
    hist["equipo_visitante"] = hist["equipo_visitante"].map(normalizar_nombre_equipo)
    hist["liga"] = LIGA_ALERTAS
    hist["temporada"] = "HISTORICO"
    hist = hist.sort_values("fecha").reset_index(drop=True)

    filas: list[dict] = []
    def fecha_local_partido(partido: dict) -> Optional[str]:
        try:
            marca = pd.Timestamp(partido["fecha"])
            if marca.tzinfo is None:
                marca = marca.tz_localize("UTC")
            return marca.tz_convert(ZONA_HORARIA_ALERTAS).date().isoformat()
        except (KeyError, TypeError, ValueError):
            return None

    eventos = [p for p in partidos if fecha_local_partido(p) == fecha_jornada]
    if not eventos:
        avisos.append(f"No hay partidos programados para {fecha_jornada} en la fuente de calendario.")

    for fixture in eventos:
        local_original, visitante_original = fixture.get("local", ""), fixture.get("visitante", "")
        local, visitante = normalizar_nombre_equipo(local_original), normalizar_nombre_equipo(visitante_original)
        evento_cuotas = next((c for c in cuotas
                              if equipos_equivalentes(c.get("home_team"), local)
                              and equipos_equivalentes(c.get("away_team"), visitante)), None)
        if evento_cuotas is None:
            avisos.append(f"{local_original} vs {visitante_original}: no se encontraron cuotas actuales.")
            continue

        try:
            fecha_partido = pd.Timestamp(fixture["fecha"])
            if fecha_partido.tzinfo is not None:
                fecha_corte = fecha_partido.tz_convert("UTC").tz_localize(None)
            else:
                fecha_corte = fecha_partido
            hist_anterior = hist[hist["fecha"] < fecha_corte].copy()
            if hist_anterior.empty:
                avisos.append(f"{local_original} vs {visitante_original}: no hay histórico anterior al partido.")
                continue
            indice = IndiceHistorico(hist_anterior)
            contexto = construir_contexto_partido(
                indice, fecha_corte, local, visitante, LIGA_ALERTAS, "HISTORICO",
                n_partidos=n_partidos, n_minimo=n_minimo, cruzar_temporadas=True,
            )
            if not (contexto.historico_local.suficiente and contexto.historico_visitante.suficiente):
                avisos.append(f"{local_original} vs {visitante_original}: histórico insuficiente para el mínimo de {n_minimo} partidos por rol.")
                continue
            referencia = promedio_referencia(contexto.historico_local.partidos, contexto.historico_visitante.partidos)
            fuerza_local = calcular_fuerza(contexto.historico_local.partidos, referencia, alpha)
            fuerza_visitante = calcular_fuerza(contexto.historico_visitante.partidos, referencia, alpha)
            goles = calcular_goles_esperados(fuerza_local, fuerza_visitante, referencia)
            probabilidades = _obtener_probabilidades(goles)
            precios = _selecciones_cuotas(evento_cuotas, local, visitante)

            for mercado, prob_modelo in probabilidades.items():
                cuota = precios.get(mercado)
                if cuota is None or pd.isna(cuota) or float(cuota) <= 1:
                    continue
                cuota = float(cuota)
                prob_implicita = 1 / cuota
                edge = prob_modelo - prob_implicita
                if edge < min_edge or edge <= 0:
                    continue
                if tipo_stake == "fijo":
                    stake = min(stake_fijo, bankroll * stake_max_pct)
                else:
                    b = cuota - 1
                    kelly = max(0.0, (prob_modelo * b - (1 - prob_modelo)) / b)
                    stake = min(bankroll * kelly * fraccion_kelly, bankroll * stake_max_pct)
                if stake <= 0:
                    continue
                filas.append({
                    "Fecha": fecha_partido,
                    "Liga": LIGA_ALERTAS,
                    "Partido": f"{local_original} vs {visitante_original}",
                    "Mercado": mercado,
                    "Prob. modelo (%)": prob_modelo * 100,
                    "Cuota": cuota,
                    "Prob. implícita (%)": prob_implicita * 100,
                    "Edge (%)": edge * 100,
                    "EV (%)": (prob_modelo * cuota - 1) * 100,
                    "Stake sugerido (u)": stake,
                    "Tipo stake": tipo_stake,
                })
        except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
            avisos.append(f"{local_original} vs {visitante_original}: no se pudo calcular el modelo ({exc}).")

    apuestas = pd.DataFrame(filas)
    if not apuestas.empty:
        apuestas = apuestas.sort_values(["Edge (%)", "EV (%)"], ascending=False).reset_index(drop=True)
    elif eventos and not avisos:
        avisos.append("No se detectaron Value Bets que superen el Edge mínimo configurado.")
    return ResultadoBoletinJornada(fecha_jornada, apuestas, avisos)


def generar_boletin_jornada(
    directorio_datos: str | Path,
    fecha: Optional[datetime] = None,
    **parametros,
) -> ResultadoBoletinJornada:
    """Consulta fixtures/cuotas actuales y genera alertas para el día local."""
    zona = ZoneInfo(ZONA_HORARIA_ALERTAS)
    fecha_local = (fecha or datetime.now(zona)).astimezone(zona).date().isoformat()
    ruta_liga = Path(directorio_datos) / LIGA_ALERTAS
    archivos = sorted(ruta_liga.glob("*.csv")) if ruta_liga.exists() else []
    if not archivos:
        return ResultadoBoletinJornada(fecha_local, pd.DataFrame(), [f"No hay CSV históricos en {ruta_liga}."])

    frames = []
    avisos = []
    for archivo in archivos:
        try:
            datos = cargar_datos(str(archivo), imprimir_resumen=False)
            if not datos.df.empty:
                frames.append(datos.df)
        except Exception as exc:
            avisos.append(f"No se pudo cargar {archivo.name}: {exc}")
    historico = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not historico.empty:
        historico = historico.drop_duplicates(
            subset=["fecha", "equipo_local", "equipo_visitante"], keep="last"
        ).sort_values("fecha").reset_index(drop=True)

    try:
        respuesta_calendario = obtener_partidos_premier()
        partidos = [
            {"local": m["homeTeam"]["name"], "visitante": m["awayTeam"]["name"], "fecha": m["utcDate"]}
            for m in respuesta_calendario.get("matches", [])
            if m.get("status") in {"SCHEDULED", "TIMED"}
            and m.get("homeTeam", {}).get("name") and m.get("awayTeam", {}).get("name")
        ]
    except Exception as exc:
        partidos = []
        avisos.append(f"No fue posible obtener el calendario: {exc}")
    try:
        cuotas = normalizar_cuotas_epl(obtener_cuotas_epl())
    except Exception as exc:
        cuotas = []
        avisos.append(f"No fue posible obtener las cuotas: {exc}")

    resultado = calcular_alertas_value_bets(partidos, cuotas, historico, fecha_local, **parametros)
    resultado.avisos = avisos + resultado.avisos
    return resultado


def boletin_markdown(resultado: ResultadoBoletinJornada, min_edge: float = 0.05) -> str:
    """Formatea la tabla priorizada como un boletín Markdown descargable."""
    lineas = [f"# Boletín de Value Bets · {resultado.fecha}", "", f"Edge mínimo: **{min_edge:.1%}**", ""]
    if resultado.apuestas.empty:
        lineas.append("No hay apuestas recomendadas para esta jornada.")
    else:
        tabla = resultado.apuestas.copy()
        for columna in ("Prob. modelo (%)", "Prob. implícita (%)", "Edge (%)", "EV (%)"):
            tabla[columna] = tabla[columna].map(lambda x: f"{x:.2f}%")
        tabla["Cuota"] = tabla["Cuota"].map(lambda x: f"{x:.2f}")
        tabla["Stake sugerido (u)"] = tabla["Stake sugerido (u)"].map(lambda x: f"{x:.2f} u")
        if "Fecha" in tabla:
            tabla["Fecha"] = pd.to_datetime(tabla["Fecha"]).dt.strftime("%H:%M UTC")
        columnas = list(tabla.columns)
        lineas.append("| " + " | ".join(columnas) + " |")
        lineas.append("| " + " | ".join(["---"] * len(columnas)) + " |")
        for fila in tabla.itertuples(index=False, name=None):
            lineas.append("| " + " | ".join(str(valor).replace("|", "\\|") for valor in fila) + " |")
    if resultado.avisos:
        lineas.extend(["", "## Avisos", *[f"- {aviso}" for aviso in resultado.avisos]])
    return "\n".join(lineas) + "\n"
