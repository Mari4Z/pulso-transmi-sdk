# Estrategia de monitoreo y reentrenamiento ante drift

Este documento responde directamente a los 6 puntos de evidencia pedidos en
la guía oficial de la fase de adaptación
(`uexternadojz/pulso-transmi/docs/fase-drift.md`). El detalle técnico completo
de cada decisión, con números antes/después, está en
[experimentos-modelos.md](experimentos-modelos.md); aquí se resume y se
enlaza.

## 1. Continuidad de la ingesta y de las submissions

- `predict.yml` envía cada 10 min, `data_collector.yml` ingiere observaciones
  cada 15 min, `drift_monitor.yml` revisa drift cada 30 min, `accuracy_monitor.yml`
  resuelve predicciones y registra el leaderboard cada 15 min, `pipeline.yml`
  reentrena cada hora como mínimo (más seguido si hay drift severo). Los
  cuatro primeros se disparan desde cron-job.org (`workflow_dispatch`), no con
  el `schedule:` nativo de GitHub Actions.
- **Por qué no el `schedule:` nativo:** se verificó en producción que GitHub
  throttlea los workflows programados de repos de bajo tráfico — un cron
  configurado cada hora, o incluso cada 30 min, terminaba corriendo cada
  5-7 horas reales (confirmado comparando `event: schedule` contra los
  timestamps reales de ejecución). Migrar los cuatro jobs a disparadores
  externos fue necesario para que la cadencia configurada fuera la cadencia
  real.
- `leaderboard_log.coverage` y `pipeline_executions.status` quedan grabados
  en cada corrida — una caída de accuracy se puede distinguir de una caída de
  cobertura (ciclos sin submission) consultando ambos campos, no solo el
  accuracy.
- **Evidencia de un fallo operativo real detectado y corregido:**
  `drift_monitor.py` tuvo un timeout silencioso (`GET /v1/observations failed:
  timed out`) que el `try/except` de nivel superior absorbía sin medir nada
  ni escribir `model_drift_log` — el job reportaba "success" a GitHub Actions
  sin haber hecho nada útil. Se agregaron 3 reintentos a esa descarga
  específicamente (commit `b629196`).

## 2. Diferenciar problemas operativos de cambios en la demanda

El caso anterior es el ejemplo concreto: antes de asumir que la demanda
dejó de tener drift porque `model_drift_log` no tenía filas nuevas, se leyó
el log de la corrida (`gh run view --log`) y se encontró el timeout — un
problema operativo, no una señal real de que el drift se detuvo. Una vez
corregido el reintento, la siguiente corrida sí midió 10 de 12 estaciones en
drift severo, confirmando que el drift seguía activo todo ese tiempo.

De forma simétrica, cuando el dashboard mostró 5 estaciones en 0% de
accuracy, se verificó primero si el dato en sí era correcto (comparando
`predictions.actual_demand` contra la observación real de la API) antes de
asumir un bug de visualización — en ese caso el 0% era real, no un error de
cálculo.

## 3. Qué dispara una evaluación o un nuevo entrenamiento, y qué datos usa

- `src/drift_monitor.py` compara la media de demanda de las últimas 24h
  contra el histórico anterior, por estación y agregado
  (`SEVERE_MEAN_CHANGE_PCT = 15.0`, test KS con `p < 0.05` además del cambio
  de magnitud). Si cualquier lectura es severa, dispara
  `pipeline.yml -f trigger=drift`.
- Además, un disparador independiente (`trigger=hourly-scheduled`, cron-job.org
  job 8550098) fuerza un intento de reentrenamiento cada hora exista o no
  drift detectado — respaldo para no depender de un solo mecanismo.
- Los datos usados son siempre observaciones descargadas en vivo desde la API
  del reto (`PulsoTransmiClient.all_observations_dataframe`), nunca una copia
  cacheada — el entrenamiento y la predicción ven exactamente lo mismo que
  está publicado al momento de correr.

## 4. Cómo se comparan versiones temporalmente, sin información futura

`train_and_evaluate()` (`src/pipeline.py`) parte el histórico en un corte
fijo: todo lo anterior a `cutoff` entrena, solo la ventana
`[cutoff, validation_end]` (últimos 7 días reales) se usa para medir —
nunca partición aleatoria, nunca datos posteriores al corte de ese
entrenamiento. El modelo candidato y el modelo activo se evalúan sobre
exactamente la misma ventana y la misma mezcla con el pronóstico estacional,
así que la comparación es una manzana contra una manzana, no una métrica
vieja contra una nueva.

## 5. Qué evidencia respalda mantener, promover o retirar una versión

Desde el commit `32c67f2`, ningún reentrenamiento se activa a ciegas:
`main()` carga el modelo activo *antes* de sobreescribirlo, lo evalúa en el
mismo holdout que el candidato nuevo, y solo promueve
(comitea `models/pulso_hgb_poisson.joblib`, activa el `model_version` en
Supabase) si el WAPE del candidato es igual o mejor. Si no, el intento queda
registrado (`training_runs`, `metrics`, `model_versions.is_active=false`,
`artifact_uri` marcado `rejected-not-committed:...`) pero el modelo activo no
cambia. Cada decisión imprime la comparación exacta en el log de la corrida,
por ejemplo:

```
Comparación vs. modelo activo: WAPE nuevo=23.86% vs. WAPE activo=23.88% -> mejora o iguala, se promueve.
```

Desde el commit `18820bf`, cada intento entrena 5 semillas distintas por
horizonte y se queda con la mejor — sube la probabilidad de encontrar una
mejora real en cada ciclo sin poder nunca empeorar el resultado (ver
`exp-20261001-best-of-seeds` en experimentos-modelos.md).

## 6. Qué ocurrió antes, durante y después de cada cambio observado

Cada experimento en [experimentos-modelos.md](experimentos-modelos.md) sigue
el mismo formato: qué se observó, qué hipótesis se probó, qué dio el
resultado medido contra datos reales (antes/después, por estación cuando
aplica), y qué se decidió. Ejemplos concretos de esta fase de drift:

- `exp-20260928-hgb-poisson-003`: colapso real de Banderas, causa
  diagnosticada (features ancladas al histórico completo), fix medido
  (WAPE 26.4%→23.4%).
- `exp-20260930-adaptive-blend` / `extended-ceiling` / `seasonal-naive`:
  mezcla con pronóstico ingenuo escalada por severidad de drift, luego
  reemplazo del ingenuo plano por uno estacional — mejora verificada en las
  12 estaciones sin excepción (79.22%→81.47% general).
- Diagnóstico explícito de overfitting (`exp-20260930-seasonal-naive`):
  se midió la brecha train/holdout para Banderas, se probó regularizar más
  fuerte para ver si la cerraba (no la cerró — descarta overfitting),
  y se buscó la causa real en otro lugar en vez de cambiar de algoritmo sin
  evidencia.

El historial de commits de este repositorio (`git log`) y las tablas
`training_runs`/`model_versions`/`metrics`/`pipeline_executions` en Supabase
son el registro verificable de cada corrida — no se cambió nunca solo el
nombre del modelo sin un reentrenamiento real detrás.
