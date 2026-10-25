"""Normalización compartida de nombres de equipos entre proveedores y CSV."""
from __future__ import annotations
import re
import unicodedata

_ALIAS_GROUPS = {
    "Arsenal": ["Arsenal FC"], "Chelsea": ["Chelsea FC"], "Liverpool": ["Liverpool FC"],
    "Manchester City": ["Manchester City FC", "Man City"], "Manchester United": ["Manchester United FC", "Man United", "Man Utd"],
    "Tottenham": ["Tottenham Hotspur", "Tottenham Hotspur FC"],
    "Newcastle": ["Newcastle United", "Newcastle United FC"],
    "Brighton": ["Brighton & Hove Albion", "Brighton & Hove Albion FC"],
    "Crystal Palace": ["Crystal Palace FC"], "Everton": ["Everton FC"], "Fulham": ["Fulham FC"],
    "Aston Villa": ["Aston Villa FC"], "Brentford": ["Brentford FC"],
    "Bournemouth": ["AFC Bournemouth", "AFC Bournemouth FC"],
    "Nottingham Forest": ["Nottingham Forest FC", "Nott'm Forest", "Nottm Forest"], "Leeds": ["Leeds United", "Leeds United FC"],
    "Sunderland": ["Sunderland AFC"], "Burnley": ["Burnley FC"],
    "Hull": ["Hull City", "Hull City AFC"], "Coventry": ["Coventry City", "Coventry City FC"],
    "Ipswich": ["Ipswich Town", "Ipswich Town FC"],
    "Wolves": ["Wolverhampton Wanderers", "Wolverhampton Wanderers FC"],
    "West Ham": ["West Ham United", "West Ham United FC"],
    "Leicester": ["Leicester City", "Leicester City FC"],
    "Atletico Madrid": ["Atlético Madrid", "Atlético de Madrid", "Atletico de Madrid"],
    "Inter": ["Internazionale", "Inter Milan"], "AC Milan": ["Milan"],
}

def _key(nombre: object) -> str:
    texto = unicodedata.normalize("NFKD", str(nombre or ""))
    texto = "".join(c for c in texto if not unicodedata.combining(c)).casefold()
    texto = re.sub(r"&", " and ", texto)
    texto = re.sub(r"\b(fc|afc|cf|sc|ac|club)\b", " ", texto)
    return re.sub(r"[^a-z0-9]+", "", texto)

_ALIASES = {_key(alias): canon for canon, aliases in _ALIAS_GROUPS.items() for alias in [canon, *aliases]}

def normalizar_nombre_equipo(nombre: object) -> str:
    """Devuelve nombre canónico conocido o el nombre desconocido saneado."""
    if nombre is None:
        return ""
    limpio = str(nombre).strip()
    if not limpio or limpio.casefold() in {"nan", "none", "null", "<na>"}:
        return ""
    return _ALIASES.get(_key(limpio), re.sub(r"\s+", " ", limpio))

def clave_equipo(nombre: object) -> str:
    return _key(normalizar_nombre_equipo(nombre))

def equipos_equivalentes(a: object, b: object) -> bool:
    return bool(clave_equipo(a)) and clave_equipo(a) == clave_equipo(b)
