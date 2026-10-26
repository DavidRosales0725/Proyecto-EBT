"""Métricas avanzadas de rendimiento y riesgo para resultados de backtesting.

Sharpe y Sortino son ratios por apuesta (no anualizados), con tasa libre de
riesgo y retorno mínimo objetivo iguales a cero. La probabilidad de ruina es
una estimación bootstrap: re-muestrea retornos y fracciones de stake observados
para un horizonte configurable, bajo independencia aproximada entre apuestas.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class MetricasRiesgo:
    sharpe: Optional[float]
    sortino: Optional[float]
    max_drawdown_unidades: Optional[float]
    max_drawdown_pct: Optional[float]
    probabilidad_ruina_pct: Optional[float]
    horizonte_ruina_apuestas: int
    simulaciones_ruina: int


def _apuestas_a_dataframe(apuestas) -> pd.DataFrame:
    if isinstance(apuestas, pd.DataFrame):
        df = apuestas.copy()
    else:
        filas = [dict(ap) if isinstance(ap, dict) else vars(ap).copy() for ap in (apuestas or [])]
        df = pd.DataFrame(filas)
    if df.empty:
        return pd.DataFrame(columns=["se_aposto", "stake", "ganancia", "cuota", "resultado_real"])
    requeridas = {"stake", "ganancia", "cuota", "resultado_real"}
    faltantes = requeridas.difference(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas para analizar apuestas: {sorted(faltantes)}")
    if "se_aposto" in df:
        df = df[df["se_aposto"].fillna(False).astype(bool)]
    else:
        df = df[df["stake"].fillna(0) > 0]
    for columna in ("stake", "ganancia", "cuota", "resultado_real"):
        df[columna] = pd.to_numeric(df[columna], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["stake", "ganancia", "cuota", "resultado_real"]
    ).query("stake > 0 and cuota > 1").copy()
    if "fecha" in df:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df = df.sort_values("fecha", kind="stable", na_position="last")
    return df.reset_index(drop=True)


def calcular_metricas_riesgo(
    apuestas,
    bankroll_inicial: float,
    *,
    horizonte_ruina_apuestas: int = 500,
    simulaciones_ruina: int = 2000,
    semilla: int = 42,
) -> MetricasRiesgo:
    """Calcula ratios por apuesta, drawdown y probabilidad bootstrap de ruina.

    La ruina se define como bankroll <= 0. Para cada simulación se re-muestrean
    pares observados de (fracción del bankroll apostada, retorno neto del stake).
    """
    if not np.isfinite(bankroll_inicial) or bankroll_inicial <= 0:
        raise ValueError("bankroll_inicial debe ser positivo y finito")
    if horizonte_ruina_apuestas < 1 or simulaciones_ruina < 1:
        raise ValueError("El horizonte y las simulaciones deben ser positivos")

    df = _apuestas_a_dataframe(apuestas)
    if df.empty:
        return MetricasRiesgo(None, None, None, None, None,
                              horizonte_ruina_apuestas, simulaciones_ruina)

    stake = df["stake"].to_numpy(dtype=float)
    ganancia = df["ganancia"].to_numpy(dtype=float)
    retornos = ganancia / stake
    desviacion = float(np.std(retornos, ddof=1)) if len(retornos) > 1 else 0.0
    sharpe = float(np.mean(retornos) / desviacion) if desviacion > 0 else None
    downside = np.minimum(retornos, 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside))))
    sortino = float(np.mean(retornos) / downside_deviation) if downside_deviation > 0 else None

    if "fecha" in df and df["fecha"].notna().all():
        grupos_fecha = df.groupby(df["fecha"].dt.normalize(), sort=True).indices.values()
        pnl_por_fecha = []
        saldo_previo = np.empty(len(df), dtype=float)
        saldo = bankroll_inicial
        for posiciones in grupos_fecha:
            saldo_previo[posiciones] = saldo
            pnl_fecha = float(ganancia[posiciones].sum())
            pnl_por_fecha.append(pnl_fecha)
            saldo += pnl_fecha
        pnl_periodo = np.asarray(pnl_por_fecha, dtype=float)
    else:
        saldo_previo = bankroll_inicial + np.concatenate(([0.0], np.cumsum(ganancia)[:-1]))
        pnl_periodo = ganancia

    equity = bankroll_inicial + np.cumsum(pnl_periodo)
    equity_con_inicio = np.concatenate(([bankroll_inicial], equity))
    picos = np.maximum.accumulate(equity_con_inicio)
    drawdowns = picos - equity_con_inicio
    drawdowns_pct = np.divide(
        drawdowns, picos, out=np.zeros_like(drawdowns), where=picos > 0
    ) * 100
    max_dd = float(np.max(drawdowns))
    max_dd_pct = float(np.max(drawdowns_pct))

    fracciones = np.divide(stake, saldo_previo, out=np.ones_like(stake), where=saldo_previo > 0)
    fracciones = np.clip(fracciones, 0.0, 1.0)
    retorno_stake = np.clip(retornos, -1.0, None)
    ya_en_ruina = bool(np.any(equity <= 0))
    if ya_en_ruina:
        prob_ruina = 100.0
    else:
        rng = np.random.default_rng(semilla)
        saldos = np.full(simulaciones_ruina, bankroll_inicial, dtype=float)
        en_ruina = np.zeros(simulaciones_ruina, dtype=bool)
        for _ in range(horizonte_ruina_apuestas):
            indices = rng.integers(0, len(df), size=simulaciones_ruina)
            multiplicadores = 1.0 + fracciones[indices] * retorno_stake[indices]
            saldos[~en_ruina] *= multiplicadores[~en_ruina]
            en_ruina |= saldos <= 0
            if en_ruina.all():
                break
        prob_ruina = float(en_ruina.mean() * 100)

    return MetricasRiesgo(
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown_unidades=max_dd,
        max_drawdown_pct=max_dd_pct,
        probabilidad_ruina_pct=prob_ruina,
        horizonte_ruina_apuestas=horizonte_ruina_apuestas,
        simulaciones_ruina=simulaciones_ruina,
    )


def desglose_por_rangos_cuota(apuestas) -> pd.DataFrame:
    """Yield, acierto, beneficio y volumen por bandas de cuota decimal."""
    etiquetas = ["<1.50", "1.50–1.80", "1.80–2.20", ">2.20"]
    columnas = ["Rango de cuota", "Apuestas", "Acierto (%)", "Yield (%)", "Beneficio (u)", "Stake total (u)"]
    df = _apuestas_a_dataframe(apuestas)
    if df.empty:
        return pd.DataFrame(columns=columnas)

    df["Rango de cuota"] = pd.cut(
        df["cuota"], bins=[-np.inf, 1.50, 1.80, 2.20, np.inf],
        labels=etiquetas, right=False, include_lowest=True,
    )
    # La banda 1.80–2.20 incluye ambos extremos; >2.20 empieza estrictamente después.
    df.loc[df["cuota"].eq(2.20), "Rango de cuota"] = etiquetas[2]
    grupos = df.groupby("Rango de cuota", observed=False, sort=False)
    resumen = grupos.agg(
        Apuestas=("ganancia", "size"),
        aciertos=("resultado_real", lambda valores: float((valores == 1).mean() * 100)),
        beneficio=("ganancia", "sum"),
        stake_total=("stake", "sum"),
    ).reset_index()
    resumen["Yield (%)"] = np.where(
        resumen["stake_total"] > 0,
        resumen["beneficio"] / resumen["stake_total"] * 100,
        np.nan,
    )
    resumen = resumen.rename(columns={
        "aciertos": "Acierto (%)", "beneficio": "Beneficio (u)", "stake_total": "Stake total (u)"
    })
    return resumen[columnas]
