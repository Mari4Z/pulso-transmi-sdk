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