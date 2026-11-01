import pandas as pd
import pytest

from motor.fixtures import FixtureProximo, normalizar_fixture, pronosticar_fixture
from motor.team_names import normalizar_nombre_equipo


def _historico_equipos():
    filas = []
    fecha = pd.Timestamp("2024-01-01")
    for n in range(24):
        local, visitante = (("Arsenal FC", "Chelsea") if n % 2 == 0 else ("Chelsea", "Arsenal"))
        filas.append((fecha, local, visitante, 1 + n % 3, n % 2))
        fecha += pd.Timedelta(days=7)
    return pd.DataFrame(filas, columns=[
        "fecha", "equipo_local", "equipo_visitante", "goles_local", "goles_visitante"
    ])


def test_normaliza_fixture_con_alias_de_columnas():
    fixture = normalizar_fixture({
        "HomeTeam": "Arsenal FC", "AwayTeam": "Chelsea FC", "Date": "2025-01-01"
    })
    assert fixture.equipo_local == "Arsenal"
    assert fixture.equipo_visitante == "Chelsea"


@pytest.mark.parametrize(("alias", "canonico"), [
    ("Ath Bilbao", "Athletic Club"),
    ("Ath Madrid", "Atletico Madrid"),
    ("Bayern München", "Bayern Munich"),
    ("BVB", "Borussia Dortmund"),
    ("FC Internazionale Milano", "Inter"),
    ("Man City", "Manchester City"),
])
def test_aliases_de_ligas_soportadas(alias, canonico):
    assert normalizar_nombre_equipo(alias) == canonico


def test_conecta_fixture_con_historico_y_ejecuta_1000_simulaciones():
    resultado = pronosticar_fixture(
        FixtureProximo("Arsenal", "Chelsea", "2025-01-01", "Premier League"),
        _historico_equipos(),
        ventana=10,
    )
    assert resultado.iteraciones == 1000
    assert resultado.partidos_local == 10
    assert resultado.partidos_visitante == 10
    assert resultado.marcador_simulado
    assert sum((resultado.victoria_local, resultado.empate, resultado.victoria_visitante)) == pytest.approx(1)


def test_fixture_sin_equipo_historico_da_aviso_controlado():
    with pytest.raises(ValueError, match="No se encontraron.*Chelsea"):
        pronosticar_fixture(
            FixtureProximo("Arsenal", "Chelsa", "2025-01-01"), _historico_equipos()
        )
