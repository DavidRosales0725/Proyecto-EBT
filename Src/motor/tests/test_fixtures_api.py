from datetime import date

import pytest
import requests

from motor.adapters.fixtures_api import (
    COMPETICIONES_GRATUITAS,
    FixturesAPIError,
    FootballDataClient,
    TIEMPO_CACHE_CLASIFICACION_SEGUNDOS,
    TIEMPO_CACHE_FIXTURES_SEGUNDOS,
    _obtener_clasificacion_cacheada,
    _obtener_fixtures_cacheado,
    normalizar_fixtures,
    normalizar_historicos,
)


class _Respuesta:
    def __init__(self, status_code, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class _SesionFalsa:
    def __init__(self, respuestas):
        self.respuestas = list(respuestas)
        self.llamadas = []

    def get(self, url, **kwargs):
        self.llamadas.append((url, kwargs))
        respuesta = self.respuestas.pop(0)
        if isinstance(respuesta, Exception):
            raise respuesta
        return respuesta


def test_normaliza_partidos_programados_y_omite_terminados():
    payload = {
        "competition": {"name": "Premier League"},
        "matches": [
            {
                "status": "TIMED",
                "utcDate": "2026-11-07T17:30:00Z",
                "homeTeam": {"name": "Arsenal FC"},
                "awayTeam": {"name": "Manchester United FC"},
            },
            {
                "status": "FINISHED",
                "utcDate": "2026-11-01T17:30:00Z",
                "homeTeam": {"name": "Chelsea FC"},
                "awayTeam": {"name": "Liverpool FC"},
            },
        ],
    }
    fixtures = normalizar_fixtures(payload, liga_historica="Premier League")
    assert len(fixtures) == 1
    assert fixtures[0].equipo_local == "Arsenal"
    assert fixtures[0].equipo_visitante == "Manchester United"
    assert fixtures[0].fecha == "2026-11-07T17:30+00:00"
    assert fixtures[0].liga == "Premier League"


def test_cliente_reintenta_429_y_envia_token_solo_en_header(monkeypatch):
    monkeypatch.setattr("motor.adapters.fixtures_api.time.sleep", lambda _segundos: None)
    sesion = _SesionFalsa([
        _Respuesta(429, headers={"Retry-After": "0"}),
        _Respuesta(200, {"matches": []}),
    ])
    cliente = FootballDataClient("secreto-de-prueba", session=sesion, max_retries=1)

    respuesta = cliente.obtener_partidos("PL", date(2026, 11, 7), date(2026, 11, 14))

    assert respuesta == {"matches": []}
    assert len(sesion.llamadas) == 2
    url, argumentos = sesion.llamadas[-1]
    assert url.endswith("/competitions/PL/matches")
    assert argumentos["params"] == {"dateFrom": "2026-11-07", "dateTo": "2026-11-14"}
    assert argumentos["headers"] == {"X-Auth-Token": "secreto-de-prueba"}
    assert "secreto-de-prueba" not in url


def test_cliente_presenta_error_util_si_token_es_rechazado():
    cliente = FootballDataClient(
        "token-invalido", session=_SesionFalsa([_Respuesta(401)]), max_retries=0
    )
    with pytest.raises(FixturesAPIError, match="rechazó el token"):
        cliente.obtener_partidos("PL", "2026-11-07", "2026-11-14")


def test_cliente_valida_codigo_y_rango_de_fechas():
    cliente = FootballDataClient("token", session=_SesionFalsa([]), max_retries=0)
    with pytest.raises(FixturesAPIError, match="Código de competición"):
        cliente.obtener_partidos("../PL", "2026-11-07", "2026-11-14")
    with pytest.raises(FixturesAPIError, match="posterior"):
        cliente.obtener_partidos("PL", "2026-11-14", "2026-11-07")


def test_cliente_consulta_temporada_finalizada():
    sesion = _SesionFalsa([_Respuesta(200, {"matches": []})])
    cliente = FootballDataClient("token", session=sesion, max_retries=0)
    assert cliente.obtener_historicos("PL", 2026) == {"matches": []}
    _url, argumentos = sesion.llamadas[0]
    assert argumentos["params"] == {"status": "FINISHED", "season": 2026}


def test_normaliza_historicos_finalizados_a_esquema_del_motor():
    payload = {"matches": [
        {"status": "FINISHED", "utcDate": "2026-10-20T19:00:00Z",
         "homeTeam": {"name": "Arsenal FC"}, "awayTeam": {"name": "Chelsea FC"},
         "score": {"fullTime": {"home": 2, "away": 1}}},
        {"status": "SCHEDULED", "utcDate": "2026-11-02T19:00:00Z",
         "homeTeam": {"name": "Chelsea FC"}, "awayTeam": {"name": "Arsenal FC"},
         "score": {"fullTime": {"home": None, "away": None}}},
    ]}
    historico = normalizar_historicos(payload)
    assert list(historico.columns) == ["fecha", "_temporada", "equipo_local", "equipo_visitante", "goles_local", "goles_visitante"]
    assert len(historico) == 1
    assert historico.iloc[0]["equipo_local"] == "Arsenal"
    assert historico.iloc[0]["goles_local"] == 2


def test_mapeo_incluye_competiciones_gratuitas_internacionales():
    assert COMPETICIONES_GRATUITAS["UEFA Champions League"]["codigo"] == "CL"
    assert COMPETICIONES_GRATUITAS["Brasileirão Série A"]["codigo"] == "BSA"
    assert "FIFA World Cup" not in COMPETICIONES_GRATUITAS
    assert "Eurocopa" not in COMPETICIONES_GRATUITAS
    assert len(COMPETICIONES_GRATUITAS) == 10


def test_consulta_clasificacion_usa_endpoint_y_season():
    sesion = _SesionFalsa([_Respuesta(200, {"standings": []})])
    cliente = FootballDataClient("token", session=sesion, max_retries=0)
    assert cliente.obtener_clasificacion("CL", 2026)["standings"] == []
    url, argumentos = sesion.llamadas[0]
    assert url.endswith("/competitions/CL/standings")
    assert argumentos["params"] == {"season": 2026}


def test_cachea_fixtures_y_clasificacion_positivos(monkeypatch):
    llamadas = {"fixtures": 0, "clasificacion": 0}

    def fixtures(self, competicion, desde, hasta):
        llamadas["fixtures"] += 1
        return {"matches": [{"status": "TIMED", "id": 1}]}

    def clasificacion(self, competicion, temporada=None):
        llamadas["clasificacion"] += 1
        return {"standings": [{"table": []}]}

    monkeypatch.setattr(FootballDataClient, "obtener_partidos", fixtures)
    monkeypatch.setattr(FootballDataClient, "obtener_clasificacion", clasificacion)
    _obtener_fixtures_cacheado.clear()
    _obtener_clasificacion_cacheada.clear()
    try:
        _obtener_fixtures_cacheado("CL", "2026-11-01", "2026-11-08", "token", "token-hash")
        _obtener_fixtures_cacheado("CL", "2026-11-01", "2026-11-08", "token", "token-hash")
        _obtener_clasificacion_cacheada("CL", 2026, "token", "token-hash")
        _obtener_clasificacion_cacheada("CL", 2026, "token", "token-hash")
        assert llamadas == {"fixtures": 1, "clasificacion": 1}
    finally:
        _obtener_fixtures_cacheado.clear()
        _obtener_clasificacion_cacheada.clear()
    assert TIEMPO_CACHE_FIXTURES_SEGUNDOS >= 6 * 3600
    assert TIEMPO_CACHE_CLASIFICACION_SEGUNDOS == 24 * 3600
