# EBT · Prediction Card

Interfaz Streamlit centrada en pronósticos rápidos de fútbol. El flujo principal selecciona una liga, una fecha y dos equipos; calcula xG, probabilidades Poisson y un marcador estimado mediante 1.000 simulaciones reproducibles.

## Ejecutar

```powershell
python -m pip install -r requirements.txt
streamlit run App/app.py
```

## Datos y ligas

Los históricos se buscan en `Data/raw/<Liga>/**/*.csv`. El selector escanea las carpetas dinámicamente y solo lista ligas con CSV que tengan esquema reconocible y partidos válidos; los archivos incompletos se explican en la interfaz. También se detectan carpetas de otras ligas añadidas bajo `Data/raw`.

La UI permite elegir **Modo Manual** o **Modo API Automática**. En automático, EBT consulta fixtures y resultados `FINISHED` de las temporadas actual y anterior en Football-Data.org; los históricos se cachean seis horas y alimentan las muestras recientes por rol. Si hay entre uno y siete partidos de la temporada actual por rol, el peso empieza en 70% actual y sube linealmente hasta 100% al octavo; la temporada anterior aporta el peso restante. Si la cobertura conjunta de API es menor de cinco partidos para algún rol, usa el CSV local cuando esté disponible. La tarjeta indica nivel de confianza, conteos por rol, fuente y fecha de actualización. Si la API falla, la interfaz ofrece el modo manual. El conector `Src/motor/fixtures.py` normaliza nombres y llama a Poisson y las 1.000 simulaciones.

## Calendarios gratuitos

El cliente opcional `Src/motor/adapters/fixtures_api.py` usa Football-Data.org API v4 para calendarios, históricos y clasificaciones. El selector usa 10 competiciones del nivel gratuito adecuadas para el flujo regular, incluida UEFA Champions League y Brasil Serie A; omite los torneos de selecciones Eurocopa y Copa del Mundo. La disponibilidad efectiva también depende del token y de la cuenta. La API limita el plan Free a 10 llamadas por minuto. EBT cachea fixtures seis horas, históricos seis horas y clasificaciones 24 horas en la caché de datos de Streamlit; reintenta fallos transitorios y las excepciones no se guardan. Consulta el [plan gratuito](https://www.football-data.org/pricing), la [cobertura](https://www.football-data.org/coverage) y la [documentación API v4](https://docs.football-data.org/general/v4/).

Para habilitarlo:

1. Crea una cuenta gratuita en Football-Data.org y genera un token.
2. Copia `.env.example` a `.env` y coloca el token en `FOOTBALL_DATA_API_KEY`, o configura `FOOTBALL_DATA_API_KEY` en el entorno. Streamlit Secrets también admite la misma clave en `.streamlit/secrets.toml`.
3. Ejecuta `streamlit run App/app.py` y selecciona **Modo API Automática**. La disponibilidad de históricos locales ya no limita las competiciones API; el plan de la cuenta puede limitar qué ligas y endpoints devuelve.

No se incluye ningún token en el repositorio. Si aún no configuras uno, o la API rechaza la consulta de fixtures o resultados finalizados, la app sigue funcionando con **Modo Manual**. El modo API requiere resultados suficientes y nombres de equipos que puedan normalizarse contra esos resultados; ante datos insuficientes, EBT informa el problema en la interfaz.

## Motor

El pronóstico usa los partidos recientes del equipo en su rol local/visitante, mezcla temporadas con ponderación exponencial (`alpha=0.90`) y limita el xG por equipo al rango `[0.05, 3.5]` para moderar proyecciones extremas. La navegación expone solo la tarjeta; las páginas anteriores de backtesting, históricos y alertas están desactivadas.
