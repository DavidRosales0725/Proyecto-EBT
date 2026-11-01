"""Normalización compartida de nombres entre proveedores y CSV históricos."""
from __future__ import annotations

import re
import unicodedata


# Se usan nombres cortos y estables para la UI. Los alias cubren nombres que
# aparecen en los CSV locales y formas habituales de football-data/Odds API.
_ALIAS_GROUPS = {
    # Premier League
    "Arsenal": ["Arsenal FC"],
    "Aston Villa": ["Aston Villa FC"],
    "Bournemouth": ["AFC Bournemouth", "AFC Bournemouth FC"],
    "Brentford": ["Brentford FC"],
    "Brighton": ["Brighton & Hove Albion", "Brighton and Hove Albion FC"],
    "Burnley": ["Burnley FC"],
    "Chelsea": ["Chelsea FC"],
    "Crystal Palace": ["Crystal Palace FC"],
    "Everton": ["Everton FC"],
    "Fulham": ["Fulham FC"],
    "Leeds": ["Leeds United", "Leeds United FC"],
    "Leicester": ["Leicester City", "Leicester City FC"],
    "Liverpool": ["Liverpool FC"],
    "Luton": ["Luton Town", "Luton Town FC"],
    "Manchester City": ["Manchester City FC", "Man City", "Man City FC"],
    "Manchester United": ["Manchester United FC", "Man United", "Man Utd", "Man United FC"],
    "Newcastle": ["Newcastle United", "Newcastle United FC"],
    "Nottingham Forest": ["Nottingham Forest FC", "Nott'm Forest", "Nottm Forest"],
    "Sheffield United": ["Sheffield United FC", "Sheff Utd"],
    "Southampton": ["Southampton FC", "Southampton AFC"],
    "Sunderland": ["Sunderland AFC", "Sunderland FC"],
    "Tottenham": ["Tottenham Hotspur", "Tottenham Hotspur FC", "Spurs"],
    "West Ham": ["West Ham United", "West Ham United FC"],
    "Wolves": ["Wolverhampton Wanderers", "Wolverhampton Wanderers FC"],
    # LaLiga
    "Alaves": ["Deportivo Alaves", "Deportivo Alavés", "Alavés"],
    "Athletic Club": ["Ath Bilbao", "Athletic Bilbao", "Athletic Club Bilbao", "Athletic Club de Bilbao"],
    "Atletico Madrid": ["Ath Madrid", "Atlético Madrid", "Atlético de Madrid", "Atletico de Madrid", "Club Atletico de Madrid"],
    "Barcelona": ["FC Barcelona", "Barcelona FC"],
    "Real Betis": ["Betis", "Real Betis Balompie", "Real Betis Balompié"],
    "Celta Vigo": ["Celta", "RC Celta", "RC Celta de Vigo", "Celta de Vigo"],
    "Espanyol": ["Espanol", "RCD Espanyol", "RCD Espanyol de Barcelona"],
    "Getafe": ["Getafe CF"],
    "Girona": ["Girona FC"],
    "Las Palmas": ["UD Las Palmas"],
    "Leganes": ["Leganés", "CD Leganes", "CD Leganés"],
    "Mallorca": ["RCD Mallorca", "Real Mallorca"],
    "Osasuna": ["CA Osasuna"],
    "Rayo Vallecano": ["Vallecano", "Rayo", "Rayo Vallecano de Madrid"],
    "Real Madrid": ["Real Madrid CF"],
    "Real Sociedad": ["Sociedad", "Real Sociedad de Futbol", "Real Sociedad de Fútbol", "La Real"],
    "Sevilla": ["Sevilla FC"],
    "Valencia": ["Valencia CF"],
    "Real Valladolid": ["Valladolid", "Real Valladolid CF"],
    "Villarreal": ["Villarreal CF"],
    # Serie A
    "Atalanta": ["Atalanta BC", "Atalanta Bergamasca Calcio"],
    "Bologna": ["Bologna FC", "Bologna FC 1909"],
    "Cagliari": ["Cagliari Calcio"],
    "Como": ["Como 1907", "Calcio Como"],
    "Empoli": ["Empoli FC"],
    "Fiorentina": ["ACF Fiorentina", "AC Fiorentina"],
    "Frosinone": ["Frosinone Calcio"],
    "Genoa": ["Genoa CFC", "Genoa Cricket and Football Club"],
    "Inter": ["Internazionale", "Inter Milan", "FC Internazionale Milano", "Internazionale Milano"],
    "Juventus": ["Juventus FC", "Juventus Turin"],
    "Lazio": ["SS Lazio", "Societa Sportiva Lazio"],
    "Lecce": ["US Lecce", "Unione Sportiva Lecce"],
    "Milan": ["AC Milan", "Associazione Calcio Milan"],
    "Monza": ["AC Monza", "Monza Brianza"],
    "Napoli": ["SSC Napoli", "Societa Sportiva Calcio Napoli"],
    "Parma": ["Parma Calcio", "Parma Calcio 1913"],
    "Roma": ["AS Roma", "Associazione Sportiva Roma"],
    "Salernitana": ["US Salernitana", "Salernitana 1919"],
    "Sassuolo": ["US Sassuolo", "Sassuolo Calcio"],
    "Torino": ["Torino FC", "Torino Football Club"],
    "Udinese": ["Udinese Calcio"],
    "Venezia": ["Venezia FC"],
    "Verona": ["Hellas Verona", "Hellas Verona FC"],
    # Bundesliga
    "Augsburg": ["FC Augsburg"],
    "Bayern Munich": ["Bayern München", "FC Bayern Munich", "FC Bayern München", "Bayern Munchen"],
    "Bochum": ["VfL Bochum", "VfL Bochum 1848"],
    "Borussia Dortmund": ["Dortmund", "BVB", "Borussia Dortmund 09"],
    "Eintracht Frankfurt": ["Ein Frankfurt", "Eintracht Frankfurt 1899"],
    "Freiburg": ["SC Freiburg", "Sport-Club Freiburg"],
    "Heidenheim": ["FC Heidenheim", "1. FC Heidenheim", "1 FC Heidenheim"],
    "Hoffenheim": ["TSG Hoffenheim", "1899 Hoffenheim"],
    "Holstein Kiel": ["Kiel", "Holstein Kiel 1900"],
    "Bayer Leverkusen": ["Leverkusen", "Bayer 04 Leverkusen", "Bayer Leverkusen 04"],
    "Mainz": ["Mainz 05", "1. FSV Mainz 05", "FSV Mainz"],
    "Borussia Monchengladbach": ["Borussia Mönchengladbach", "M'gladbach", "Monchengladbach", "Gladbach"],
    "RB Leipzig": ["RasenBallsport Leipzig", "Red Bull Leipzig"],
    "St Pauli": ["FC St Pauli", "FC St. Pauli", "St. Pauli"],
    "Stuttgart": ["VfB Stuttgart"],
    "Union Berlin": ["1. FC Union Berlin", "FC Union Berlin"],
    "Werder Bremen": ["SV Werder Bremen", "Werder"],
    "Wolfsburg": ["VfL Wolfsburg"],
}


def _key(nombre: object) -> str:
    texto = "" if nombre is None else str(nombre)
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c)).casefold()
    texto = texto.replace("&", " and ")
    texto = re.sub(r"\b(fc|afc|cf|sc|ac|club)\b", " ", texto)
    return re.sub(r"[^a-z0-9]+", "", texto)


_ALIASES = {
    _key(alias): canon
    for canon, aliases in _ALIAS_GROUPS.items()
    for alias in (canon, *aliases)
}


def normalizar_nombre_equipo(nombre: object) -> str:
    """Devuelve un nombre canónico conocido o uno desconocido saneado."""
    if nombre is None:
        return ""
    limpio = str(nombre).strip()
    if not limpio or limpio.casefold() in {"nan", "none", "null", "<na>"}:
        return ""
    return _ALIASES.get(_key(limpio), re.sub(r"\s+", " ", limpio))


def clave_equipo(nombre: object) -> str:
    """Clave sin acentos, puntuación ni sufijos de club para cruces exactos."""
    return _key(normalizar_nombre_equipo(nombre))


def equipos_equivalentes(a: object, b: object) -> bool:
    return bool(clave_equipo(a)) and clave_equipo(a) == clave_equipo(b)
