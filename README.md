# EBT · Prediction Card

Interfaz Streamlit centrada en pronósticos rápidos de fútbol. El flujo principal selecciona una liga, una fecha y dos equipos; calcula xG, probabilidades Poisson y un marcador estimado mediante 1.000 simulaciones reproducibles.

## Ejecutar

```powershell
python -m pip install -r requirements.txt
streamlit run App/app.py
```

## Datos y ligas

Los históricos se buscan en `Data/raw/<Liga>/**/*.csv`. El selector escanea las carpetas dinámicamente y solo lista ligas con CSV que tengan esquema reconocible y partidos válidos; los archivos incompletos se explican en la interfaz. También se detectan carpetas de otras ligas añadidas bajo `Data/raw`.

La UI permite elegir **Modo Manual** o **Modo API Automática**. En automático, EBT consulta tanto los fixtures próximos como los partidos `FINISHED` de la temporada actual en Football-Data.org; los históricos API se cachean seis horas y alimentan directamente las muestras recientes por rol, sin leer los CSV locales. Si la API no está configurada o falla, la interfaz muestra un aviso y ofrece el modo manual con CSV. El conector `Src/motor/fixtures.py` acepta objetos internos y registros con columnas habituales (`HomeTeam`/`AwayTeam`, `home_team`/`away_team`), normaliza nombres y llama a Poisson y las 1.000 simulaciones.

## Calendarios gratuitos

El cliente opcional `Src/motor/adapters/fixtures_api.py` usa Football-Data.org API v4 para calendarios. El plan Free publica fixtures con demora, cubre una selección de competiciones y limita el acceso a 10 consultas por minuto; la cobertura puede cambiar, así que comprueba el [plan gratuito](https://www.football-data.org/pricing), la [cobertura](https://www.football-data.org/coverage) y la [documentación de la API](https://www.football-data.org/documentation/api). EBT cachea respuestas correctas durante una hora, reintenta errores transitorios unas pocas veces y no cachea excepciones.

Para habilitarlo:

1. Crea una cuenta gratuita en Football-Data.org y genera un token.
2. Copia `.env.example` a `.env` y coloca el token en `FOOTBALL_DATA_API_KEY`, o configura `FOOTBALL_DATA_API_KEY` en el entorno. Streamlit Secrets también admite la misma clave en `.streamlit/secrets.toml`.
3. Ejecuta `streamlit run App/app.py` y selecciona **Modo API Automática**. La disponibilidad de históricos locales ya no limita las competiciones API; el plan de la cuenta puede limitar qué ligas y endpoints devuelve.

No se incluye ningún token en el repositorio. Si aún no configuras uno, o la API rechaza la consulta de fixtures o resultados finalizados, la app sigue funcionando con **Modo Manual**. El modo API requiere resultados suficientes y nombres de equipos que puedan normalizarse contra esos resultados; ante datos insuficientes, EBT informa el problema en la interfaz.

## Motor

El pronóstico usa los partidos recientes del equipo en su rol local/visitante, ajusta goles por fuerza de ataque/defensa de los rivales previos y aplica ponderación exponencial (`alpha=0.90`). La navegación expone solo la tarjeta; las páginas anteriores de backtesting, históricos y alertas están desactivadas.
