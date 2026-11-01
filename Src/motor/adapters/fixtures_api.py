"""Cliente acotado para fixtures de Football-Data.org API v4.

El cliente se limita a partidos programados. No consulta cuotas ni escribe el
token en disco. En Streamlit las respuestas correctas se cachean por fecha,
competición y huella del token; las excepciones nunca se cachean.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import os
from pathlib import Path
import re
import time
from typing import Any, Mapping

import requests

try:
    import streamlit as st
except ModuleNotFoundError:  # permite usar el cliente desde scripts del motor
    st = None

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

from ..fixtures import FixtureProximo
from ..team_names import normalizar_nombre_equipo


_ROOT = Path(__file__).resolve().parents[3]
if load_dotenv is not None:
    load_dotenv(_ROOT / ".env", override=False)

BASE_URL = "https://api.football-data.org/v4"
TIEMPO_CACHE_SEGUNDOS = 3600
TIEMPO_CACHE_HISTORICOS_SEGUNDOS = 6 * 3600
COMPETICIONES_GRATUITAS = {
    "Premier League": {"codigo": "PL", "historico": "Premier League"},
    "LaLiga": {"codigo": "PD", "historico": "LaLiga"},
    "Serie A": {"codigo": "SA", "historico": "Serie A"},
    "Bundesliga": {"codigo": "BL1", "historico": "Bundesliga"},
    "Ligue 1": {"codigo": "FL1", "historico": "Ligue 1"},
    "Championship": {"codigo": "ELC", "historico": "Championship"},
    "Eredivisie": {"codigo": "DED", "historico": "Eredivisie"},
    "Primeira Liga": {"codigo": "PPL", "historico": "Primeira Liga"},
}


class FixturesAPIError(RuntimeError):
    """Fallo de configuración, red, cuota o formato de respuesta de fixtures."""


def leer_token_api() -> str | None:
    """Lee el token desde entorno/.env o Streamlit secrets, sin registrarlo."""
    token = os.getenv("FOOTBALL_DATA_API_KEY", "").strip()
    if token:
        return token
    if st is not None:
        try:
            return str(st.secrets.get("FOOTBALL_DATA_API_KEY", "")).strip() or None
        except Exception:
            # Fuera del runtime o sin secrets.toml: el modo manual sigue libre.
            return None
    return None


class FootballDataClient:
    """Cliente HTTP pequeño con timeout, reintentos y mensajes saneados."""

    def __init__(
        self,
        token: str,
        *,
        session: requests.Session | None = None,
        timeout: float = 12.0,
        max_retries: int = 2,
        backoff_seconds: float = 0.6,
    ) -> None:
        if not token or not token.strip():
            raise FixturesAPIError("Falta FOOTBALL_DATA_API_KEY; configura el token para habilitar el modo API.")
        if max_retries < 0:
            raise ValueError("max_retries no puede ser negativo")
        self._token = token.strip()
        self._session = session or requests.Session()
        self._timeout = timeout
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds

    def obtener_partidos(
        self,
        competicion: str,
        fecha_desde: date | str,
        fecha_hasta: date | str,
    ) -> dict[str, Any]:
        codigo = competicion.strip().upper()
        if not re.fullmatch(r"[A-Z0-9]{2,5}", codigo):
            raise FixturesAPIError("Código de competición inválido.")
        desde, hasta = _fecha_iso(fecha_desde), _fecha_iso(fecha_hasta)
        if hasta < desde:
            raise FixturesAPIError("La fecha final debe ser igual o posterior a la inicial.")

        url = f"{BASE_URL}/competitions/{codigo}/matches"
        params = {"dateFrom": desde, "dateTo": hasta}
        headers = {"X-Auth-Token": self._token}
        ultimo_error: Exception | None = None

        for intento in range(self._max_retries + 1):
            try:
                respuesta = self._session.get(
                    url, params=params, headers=headers, timeout=self._timeout
                )
            except requests.RequestException as exc:
                ultimo_error = exc
                if intento < self._max_retries:
                    time.sleep(self._backoff_seconds * (2 ** intento))
                    continue
                raise FixturesAPIError("No se pudo conectar con Football-Data.org; se puede continuar en modo manual.") from None

            if respuesta.status_code == 429 or 500 <= respuesta.status_code < 600:
                if intento < self._max_retries:
                    time.sleep(_espera_reintento(respuesta, intento, self._backoff_seconds))
                    continue
            if respuesta.status_code in (401, 403):
                raise FixturesAPIError("Football-Data.org rechazó el token o la competición no está habilitada en el plan de la cuenta.")
            if respuesta.status_code == 429:
                raise FixturesAPIError("Se alcanzó el límite temporal de Football-Data.org. Reintenta más tarde o usa modo manual.")
            try:
                respuesta.raise_for_status()
            except requests.HTTPError:
                raise FixturesAPIError(f"Football-Data.org respondió con HTTP {respuesta.status_code}.") from None
            try:
                payload = respuesta.json()
            except (ValueError, requests.JSONDecodeError):
                raise FixturesAPIError("Football-Data.org devolvió una respuesta JSON inválida.") from None
            if not isinstance(payload, dict) or not isinstance(payload.get("matches"), list):
                raise FixturesAPIError("La respuesta de Football-Data.org no contiene una lista de partidos válida.")
            return payload

        raise FixturesAPIError("No se pudo obtener el calendario.") from ultimo_error

    def obtener_historicos(self, competicion: str, temporada: int | None = None) -> dict[str, Any]:
        """Obtiene los resultados FINALIZADOS de la temporada actual (año inicial)."""
        codigo = competicion.strip().upper()
        if not re.fullmatch(r"[A-Z0-9]{2,5}", codigo):
            raise FixturesAPIError("Código de competición inválido.")
        temporada = temporada or _temporada_actual(codigo)
        if not isinstance(temporada, int) or not 2000 <= temporada <= 2100:
            raise FixturesAPIError("Temporada inválida.")
        url = f"{BASE_URL}/competitions/{codigo}/matches"
        ultimo_error: Exception | None = None
        for intento in range(self._max_retries + 1):
            try:
                respuesta = self._session.get(
                    url, params={"status": "FINISHED", "season": temporada},
                    headers={"X-Auth-Token": self._token}, timeout=self._timeout,
                )
            except requests.RequestException as exc:
                ultimo_error = exc
                if intento < self._max_retries:
                    time.sleep(self._backoff_seconds * (2 ** intento))
                    continue
                raise FixturesAPIError("No se pudo conectar para cargar resultados históricos; usa modo manual.") from None
            if respuesta.status_code == 429 or 500 <= respuesta.status_code < 600:
                if intento < self._max_retries:
                    time.sleep(_espera_reintento(respuesta, intento, self._backoff_seconds))
                    continue
            if respuesta.status_code in (401, 403):
                raise FixturesAPIError("Football-Data.org rechazó el token o el acceso a resultados no está habilitado.")
            if respuesta.status_code == 429:
                raise FixturesAPIError("Se alcanzó el límite temporal de Football-Data.org. Reintenta más tarde o usa modo manual.")
            try:
                respuesta.raise_for_status()
            except requests.HTTPError:
                raise FixturesAPIError(f"Football-Data.org respondió con HTTP {respuesta.status_code} al cargar resultados.") from None
            try:
                payload = respuesta.json()
            except (ValueError, requests.JSONDecodeError):
                raise FixturesAPIError("Football-Data.org devolvió resultados en JSON inválido.") from None
            if not isinstance(payload, dict) or not isinstance(payload.get("matches"), list):
                raise FixturesAPIError("La respuesta histórica no contiene una lista de partidos válida.")
            return payload
        raise FixturesAPIError("No se pudieron obtener resultados históricos.") from ultimo_error


def _temporada_actual(codigo: str, hoy: date | None = None) -> int:
    """Año de inicio: temporada europea julio-junio, MLS año natural."""
    hoy = hoy or date.today()
    return hoy.year if codigo == "MLS" or hoy.month >= 7 else hoy.year - 1


def _fecha_iso(valor: date | str) -> str:
    try:
        if isinstance(valor, datetime):
            return valor.date().isoformat()
        return date.fromisoformat(valor).isoformat() if isinstance(valor, str) else valor.isoformat()
    except (TypeError, ValueError):
        raise FixturesAPIError("Fecha de consulta inválida; usa el formato AAAA-MM-DD.") from None


def _espera_reintento(respuesta, intento: int, backoff: float) -> float:
    retry_after = getattr(respuesta, "headers", {}).get("Retry-After")
    if retry_after:
        try:
            return min(max(float(retry_after), 0.0), 10.0)
        except ValueError:
            try:
                fecha_reintento = datetime.strptime(retry_after, "%a, %d %b %Y %H:%M:%S GMT").replace(tzinfo=timezone.utc)
                return min(max((fecha_reintento - datetime.now(timezone.utc)).total_seconds(), 0.0), 10.0)
            except ValueError:
                pass
    return min(backoff * (2 ** intento), 10.0)


def normalizar_fixtures(
    payload: Mapping[str, Any],
    *,
    liga_historica: str | None = None,
) -> tuple[FixtureProximo, ...]:
    """Convierte partidos API pendientes en objetos compatibles con el motor."""
    partidos = payload.get("matches", [])
    if not isinstance(partidos, list):
        raise FixturesAPIError("La respuesta de fixtures tiene un formato inesperado.")

    resultado: list[FixtureProximo] = []
    for partido in partidos:
        if not isinstance(partido, Mapping):
            continue
        if partido.get("status") not in ("SCHEDULED", "TIMED"):
            continue
        local = partido.get("homeTeam")
        visitante = partido.get("awayTeam")
        fecha = partido.get("utcDate")
        if not isinstance(local, Mapping) or not isinstance(visitante, Mapping) or not fecha:
            continue
        nombre_local = normalizar_nombre_equipo(local.get("name"))
        nombre_visitante = normalizar_nombre_equipo(visitante.get("name"))
        if not nombre_local or not nombre_visitante:
            continue
        try:
            fecha_partido = datetime.fromisoformat(str(fecha).replace("Z", "+00:00"))
            if fecha_partido.tzinfo is None:
                fecha_partido = fecha_partido.replace(tzinfo=timezone.utc)
            fecha_normalizada = fecha_partido.astimezone(timezone.utc).isoformat(timespec="minutes")
        except ValueError:
            continue
        competencia = partido.get("competition") or payload.get("competition") or {}
        liga = (
            liga_historica
            or (competencia.get("name", "") if isinstance(competencia, Mapping) else "")
        )
        resultado.append(FixtureProximo(nombre_local, nombre_visitante, fecha_normalizada, str(liga)))
    return tuple(resultado)


def normalizar_historicos(payload: Mapping[str, Any]):
    """Convierte resultados finalizados a columnas canónicas del predictor."""
    import pandas as pd

    partidos = payload.get("matches", [])
    if not isinstance(partidos, list):
        raise FixturesAPIError("La respuesta histórica tiene un formato inesperado.")
    filas = []
    for partido in partidos:
        if not isinstance(partido, Mapping) or partido.get("status") != "FINISHED":
            continue
        local, visitante = partido.get("homeTeam"), partido.get("awayTeam")
        score = partido.get("score") or {}
        full_time = score.get("fullTime", {}) if isinstance(score, Mapping) else {}
        if not isinstance(local, Mapping) or not isinstance(visitante, Mapping):
            continue
        try:
            fecha = pd.Timestamp(partido.get("utcDate"))
            goles_local, goles_visitante = int(full_time.get("home")), int(full_time.get("away"))
        except (TypeError, ValueError, OverflowError):
            continue
        nombre_local = normalizar_nombre_equipo(local.get("name"))
        nombre_visitante = normalizar_nombre_equipo(visitante.get("name"))
        if pd.isna(fecha) or not nombre_local or not nombre_visitante or goles_local < 0 or goles_visitante < 0:
            continue
        if fecha.tzinfo is not None:
            fecha = fecha.tz_convert(None)
        filas.append({"fecha": fecha, "equipo_local": nombre_local,
                      "equipo_visitante": nombre_visitante, "goles_local": goles_local,
                      "goles_visitante": goles_visitante})
    columnas = ["fecha", "equipo_local", "equipo_visitante", "goles_local", "goles_visitante"]
    return pd.DataFrame(filas, columns=columnas).sort_values("fecha", kind="stable").reset_index(drop=True)


def _obtener_fixtures_cacheado(
    competicion: str,
    fecha_desde: str,
    fecha_hasta: str,
    _token: str,
    huella_token: str,
) -> dict[str, Any]:
    del huella_token  # forma parte de la clave de Streamlit, el secreto no
    return FootballDataClient(_token).obtener_partidos(competicion, fecha_desde, fecha_hasta)


def _obtener_historicos_cacheado_impl(competicion: str, temporada: int, _token: str, huella_token: str):
    del huella_token
    return FootballDataClient(_token).obtener_historicos(competicion, temporada)


if st is not None:
    _obtener_fixtures_cacheado = st.cache_data(
        ttl=TIEMPO_CACHE_SEGUNDOS, show_spinner=False
    )(_obtener_fixtures_cacheado)
    _obtener_historicos_cacheado = st.cache_data(
        ttl=TIEMPO_CACHE_HISTORICOS_SEGUNDOS, show_spinner=False
    )(_obtener_historicos_cacheado_impl)
else:
    _obtener_historicos_cacheado = _obtener_historicos_cacheado_impl


def obtener_fixtures_proximos(
    competicion: str,
    fecha_desde: date | str,
    fecha_hasta: date | str,
    *,
    token: str | None = None,
) -> tuple[FixtureProximo, ...]:
    """Consulta y normaliza el calendario; usa caché positiva de una hora."""
    api_token = (token or leer_token_api() or "").strip()
    if not api_token:
        raise FixturesAPIError("FOOTBALL_DATA_API_KEY no está configurado; se usa el modo manual.")
    desde, hasta = _fecha_iso(fecha_desde), _fecha_iso(fecha_hasta)
    huella = hashlib.sha256(api_token.encode("utf-8")).hexdigest()
    payload = _obtener_fixtures_cacheado(competicion, desde, hasta, api_token, huella)
    return normalizar_fixtures(payload)


def obtener_historicos_temporada(competicion: str, *, temporada: int | None = None, token: str | None = None):
    """Históricos de temporada canónicos; la respuesta correcta vive seis horas en caché."""
    api_token = (token or leer_token_api() or "").strip()
    if not api_token:
        raise FixturesAPIError("FOOTBALL_DATA_API_KEY no está configurado.")
    temporada = temporada or _temporada_actual(competicion.strip().upper())
    huella = hashlib.sha256(api_token.encode("utf-8")).hexdigest()
    payload = _obtener_historicos_cacheado(competicion, temporada, api_token, huella)
    return normalizar_historicos(payload)
