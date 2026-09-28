# Experimentos de modelos

## Experimento seleccionado

- Experimento: `exp-20260918-hgb-poisson-001`
- Modelo: `hgb-poisson-v1`
- Artefacto: `models/pulso_hgb_poisson.joblib`
- Objetivo: predecir demanda por estación a 15, 30, 45 y 60 minutos.
- Tipo: aprendizaje supervisado, un modelo por horizonte.
- Dataset: `pulso-transmi-starter-v1`.

El entrenamiento usa las observaciones anteriores a los últimos siete días. La
validación usa exclusivamente esos siete días finales, sin partición aleatoria.
Las variables incluyen lags, medias móviles, calendario local de Bogotá,
estación, corredor, coordenadas, clima y eventos.

## Comparación

| Modelo | Pérdida | WAPE medio | Accuracy media | MAE medio |
|---|---|---:|---:|---:|
| `HistGradientBoostingRegressor` | Poisson | **12,743%** | **87,257%** | **46,218** |
| `CatBoostRegressor` | MAE | 12,911% | 87,089% | 46,829 |
| `HistGradientBoostingRegressor` | Absolute error | 12,962% | 87,038% | 47,014 |
| Baseline lag 24h | N/A | 20,997% | 79,003% | 76,156 |

El modelo Poisson fue el mejor en los cuatro horizontes:

| Horizonte | WAPE | Accuracy |
|---:|---:|---:|
| 15 min | 12,504% | 87,496% |
| 30 min | 12,659% | 87,341% |
| 45 min | 12,837% | 87,163% |
| 60 min | 12,972% | 87,028% |

Nota: estas cifras del 18-sep se midieron sobre un holdout que terminaba en
`2026-09-09` (el límite del histórico estático de entonces). El experimento
siguiente las repite sobre un holdout genuinamente más reciente.

## exp-20260927-hgb-poisson-002: búsqueda de hiperparámetros + features

Con el pipeline de predicción ya corrigiendo el bug de datos congelados (ver
commits `d040fb9`/`31aa0d1`), se hizo una búsqueda de hiperparámetros contra
datos en vivo para ver si quedaba margen de mejora. Metodología: mismo split
que `src/pipeline.py::train_and_evaluate` (train < cutoff, holdout = últimos
7 días reales), accuracy = 1 − WAPE promediado entre los 4 horizontes.

- **Ronda 1** (12 combinaciones de `learning_rate`/`max_leaf_nodes`/
  `l2_regularization`/`max_iter` alrededor del campeón anterior): resultados
  apretados entre 83.0% y 83.3% — el modelo no estaba limitado por
  hiperparámetros.
- Se probó además quitar `rain_mm`, `rain_forecast`, `temperature_c`,
  `temperature_forecast`, `event_intensity` de las features: **83.58%**, el
  mejor resultado de la ronda. Motivo: `context` no tiene stream en vivo (ver
  `PulsoTransmiClient.stream_observations_dataframe`), así que en predicción
  real esas columnas son siempre el último valor conocido repetido hacia
  adelante — señal vieja, no información real del momento.
- **Ronda 2** (10 combinaciones más, ya sin esas features): mejor resultado
  **83.69%** con `learning_rate=0.08, max_leaf_nodes=63,
  l2_regularization=2.0, max_iter=300`.

| Configuración | Accuracy holdout (promedio 4 horizontes) |
|---|---:|
| Campeón anterior (`exp-...-001`, con clima/eventos) | 83.14% |
| Mejor de la ronda 1 (con clima/eventos) | 83.25% |
| **Mejor de la ronda 2 (sin clima/eventos) — adoptado** | **83.69%** |

Ganancia modesta (+0.55pp) pero consistente en dos búsquedas independientes,
y el modelo queda más simple (17 features numéricas en vez de 22, sin
depender de una señal que en producción siempre está desactualizada).
`src/pipeline.py::HYPERPARAMETERS`/`NUMERIC_FEATURES` ya reflejan esta
configuración; `pulso_transmi.pipeline._prepare_matrix` no necesitó cambios
porque reindexa a las columnas que trajo el modelo activo.

Los 23 runs de ambas rondas quedaron loggeados en MLflow
(`mlruns_search/`, experimento `pulso-transmi-hgb-poisson-search` — no
committeado, es un experimento exploratorio de una sola vez, no el registro
de producción en `mlruns/`).

## exp-20260928-hgb-poisson-003: quitar promedios lentos por estación

Motivado por una pregunta puntual: la estación Banderas (`05100`) predecía
muy mal (accuracy 4-15% en predicciones reales resueltas, vs. 60-74% en
las otras 11 estaciones). La causa: Banderas tuvo una caída real de
demanda de ~47% (~592 → ~315 pasajeros/15min) a partir del 13-sep, y
`station_mean`/`station_hour_mean`/`station_weekday_hour_mean` son
promedios sobre *todo* el histórico de entrenamiento — con 45 días de
histórico estático más un stream que recién empieza a crecer, esas
features tardan mucho en "olvidar" el nivel viejo, así que seguían
ancladas a ~592 mientras la demanda real ya estaba en ~315.

Dos enfoques probados contra el mismo holdout, midiendo tanto el WAPE
general como el de Banderas específicamente en los 2 días posteriores a
la caída (donde el efecto es más agudo):

| Enfoque | WAPE general (foco) | Accuracy Banderas (foco) |
|---|---:|---:|
| Actual (promedios sobre todo el histórico) | 26.4% | 0.0% |
| Ventana de 3-21 días para esas 3 features | ~26.0-26.3% | 0.0% (sin cambio) |
| **Quitar las 3 features por completo — adoptado** | **23.4%** | **23.5%** |

Ventanear no funcionó: con una ventana corta, muchas combinaciones
estación×día-de-semana×hora quedan sin ninguna observación, así que esas
filas de entrenamiento se descartan por completo (arrastra el problema en
vez de resolverlo). Quitar las features por completo sí ayuda, y en las
dos dimensiones a la vez — el modelo se queda con `lag_1/4/16/96/672` y
`rolling_mean_4/16/96/672`, que ya reaccionan al nivel de demanda actual
sin depender de un promedio histórico lento.

`NUMERIC_FEATURES` en `src/pipeline.py` ya no incluye esas 3 columnas.
`add_training_statistics()` se mantiene solo porque `src/inference.py`
todavía la importa; no tiene efecto en el modelo entrenado porque esas
columnas ya no están en `NUMERIC_FEATURES`. Tampoco requirió cambios en
`pulso_transmi/pipeline.py` — `_prepare_matrix` reindexa a las columnas
del modelo activo, igual que en `exp-...-002`.

## Reproducibilidad

Regenerar el modelo desde el API:

```bash
python -m pip install -e '.[ml]'
PYTHONPATH=src python examples/03_train_poisson_model.py
```

El script guarda el bundle joblib y un JSON con el dataset, cutoff,
hiperparámetros, columnas, versiones de librerías y número de filas usadas por
horizonte. El registro completo del benchmark está en
`experiments/exp-20260918-model-benchmark.json`.

## Submission

El modelo fue enviado al ciclo de práctica mediante la ruta oficial
`POST /v1/submissions`. La guía reproducible del payload, autenticación y
verificación está en [submissions.md](submissions.md).

- Submission: `sub_ac85fb2075d84c9b9a7c4c22662ab669`
- Estado: `accepted`
- Predicciones recibidas: `12`

## Reentrenamiento automatizado y MLflow

`src/pipeline.py` (workflow `pulso-transmi-pipeline`) reentrena el modelo con
los mismos hiperparámetros de `exp-20260918-hgb-poisson-001`, usando el
histórico estático + el stream en vivo (`PulsoTransmiClient.all_observations_dataframe`).
Cada corrida:

1. Entrena con todo lo anterior al holdout y evalúa en los últimos 7 días
   reales (accuracy = 1 − WAPE, por horizonte).
2. Registra el experimento en MLflow — tracking local en `mlruns/` (sin
   servidor separado), commiteado a git junto con el modelo. Para verlo:

   ```bash
   pip install -r requirements.txt
   mlflow ui --backend-store-uri file:mlruns
   ```

   Abre `http://127.0.0.1:5000` y ahí está el historial completo:
   hiperparámetros, ventanas de train/validation, accuracy/WAPE por
   horizonte de cada reentrenamiento.
3. Registra el mismo resultado en Supabase (`training_runs`, `model_versions`,
   `metrics`) para que el dashboard y el resto del pipeline lo vean sin
   depender de MLflow.

Se dispara manualmente (`gh workflow run pipeline.yml -f trigger=manual`) o
automáticamente cuando `drift_monitor.py` detecta drift severo
(`trigger=drift`, ver `SEVERE_MEAN_CHANGE_PCT` en ese archivo).