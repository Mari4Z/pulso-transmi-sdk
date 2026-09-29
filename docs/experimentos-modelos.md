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

## exp-20260928-hgb-poisson-004: features de tendencia reciente

Con exp-003 ya en producción, se buscó margen adicional de mejora
comparando 5 variantes de features contra el mismo holdout (últimos 7
días reales, accuracy = 1 − WAPE promediado entre los 4 horizontes),
partiendo del set de exp-003 (sin `station_mean`/`station_hour_mean`/
`station_weekday_hour_mean`, sin clima/eventos):

| Variante | Accuracy general | Accuracy Banderas (05100) |
|---|---:|---:|
| Baseline (exp-003, en producción) | 81.78% | 75.39% |
| + `rolling_std_4/16/96` (volatilidad) | 81.94% | 76.12% |
| + `hour_sin`/`hour_cos` (hora cíclica) | 81.83% | 75.02% |
| **+ `trend_16`/`trend_96` — adoptado** | **82.38%** | **77.09%** |
| + las tres combinadas | 82.40% | 77.31% |

`trend_16`/`trend_96` son `lag_1 − rolling_mean_16` y
`lag_1 − rolling_mean_96`: en vez de dejar que el modelo infiera por su
cuenta qué tan lejos está la lectura actual de su nivel reciente (con
`lag_1` y `rolling_mean_*` como columnas separadas), se lo entrega ya
calculado. La ganancia es consistente en las dos dimensiones que importan
— general y, más pronunciada, en Banderas — así que complementa el fix de
exp-003 en vez de duplicarlo. `rolling_std_*` y `hour_sin`/`hour_cos` no
superan el ruido por sí solos (incluso `hour_sin`/`hour_cos` empeora
Banderas ligeramente) y combinarlos con `trend_*` no suma más allá del
margen de error (82.38% → 82.40%), así que se dejaron fuera para no
sumar columnas sin beneficio claro.

Se repitió además la búsqueda de hiperparámetros (6 combinaciones
alrededor del campeón actual) sobre este set con `trend_*` ya incluido:
resultados apretados entre 82.28% y 82.41% — confirma lo ya visto en
`exp-...-002`, el modelo no está limitado por hiperparámetros, así que
`HYPERPARAMETERS` en `src/pipeline.py` no cambió.

`NUMERIC_FEATURES` en `src/pipeline.py` y `build_features()` en
`pulso_transmi/pipeline.py` (usado en predicción real, no solo en
entrenamiento) ambos calculan ahora `trend_16`/`trend_96` — necesario
para que `_prepare_matrix` no las rellene con `0.0` al no encontrarlas.

## exp-20260929-shock-blend: mezcla con pronóstico ingenuo ante shocks

Motivado por una caída real de posición en el leaderboard (rolling_24h
78.6%→71.2%, puesto 14→19 en ~24h) mientras el líder se mantenía en 88%.
La causa: el reloj virtual de la competencia avanzó (fase de adaptación)
y trajo un shock de demanda simultáneo en 4 estaciones — Banderas
(`05100`, -69%), `03000` (-24%), `05000` (+155%) y `02300` (+149%) — en
las últimas ~18h de datos. `drift_monitor.py` lo detectó y disparó un
reentrenamiento automático correctamente, pero incluso el modelo recién
reentrenado predice mal justo después de un salto: apenas tiene un
puñado de filas del nuevo nivel de demanda para entrenar.

Se comparó, sobre el mismo holdout de 7 días, el modelo actual contra un
pronóstico ingenuo puro (`lag_1`, el último valor observado) y varias
mezclas, midiendo tanto el accuracy general como el de una ventana de
"shock" (últimas 18h, donde vive el evento):

| Peso del ingenuo (`lag_1`) | Accuracy holdout completo | Accuracy ventana de shock |
|---:|---:|---:|
| 0.0 (modelo puro, anterior) | 80.64% | 63.69% |
| 0.1 | 81.35-81.49% | 66.61-67.90% |
| **0.2** | **81.50%** (máximo) | 69.07-70.13% |
| **0.25 — adoptado** | ~81.3% | ~70.6% |
| 0.3 | 81.12% | 71.05% |
| 0.5 | 79.04% | 73.64% |
| 0.7 | 75.59% | 74.35% |

El pronóstico ingenuo no tiene el retraso estructural del modelo: *es*
literalmente el nivel más reciente, así que reacciona a un salto de
demanda al instante, mientras el modelo tiene que esperar a acumular
filas del nuevo régimen y reentrenar. Mezclar una fracción moderada
(`0.25`) del ingenuo cuesta menos de 0.2pp en datos tranquilos (el 0.2
es el óptimo exacto del holdout completo, pero la curva es casi plana
entre 0.1 y 0.3) y recupera varios puntos justo cuando más importa: en
la ventana de shock, 0.25 ronda 70-71% de accuracy frente al 63.69% del
modelo puro, sin necesitar saber de antemano qué estación está en shock.

Antes de esto se probó también blindar la mezcla combinándola solo con
el modelo (sin ingenuo) contra el shock, y no alcanzaba: el modelo puro
predijo 0% de accuracy en Banderas en esa ventana porque seguía anclado
al nivel viejo con apenas una fracción de filas del nuevo.

`NAIVE_BLEND_WEIGHT = 0.25` se agregó en `pulso_transmi/pipeline.py`
(`predict_targets`, donde se genera la predicción real que se envía) y
en `src/pipeline.py` (`train_and_evaluate`, para que las métricas de
holdout reporten lo mismo que efectivamente se somete). No requirió
cambios en el modelo entrenado ni en `NUMERIC_FEATURES` — es un ajuste
en tiempo de predicción, no una feature nueva.

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