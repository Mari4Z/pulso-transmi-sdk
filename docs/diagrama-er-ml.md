# Modelo entidad-relación para Machine Learning

El modelo conserva los datos fuente sin sobrescribirlos y separa la generación
de features, los experimentos, los modelos publicados, las predicciones y el
monitoreo. Todas las entidades temporales usan timestamps con zona horaria.

```mermaid
erDiagram
    STATION ||--o{ OBSERVATION : has
    OBSERVATION }o--|| CONTEXT : "at timestamp"
    STATION ||--o{ FEATURE_VECTOR : receives
    FEATURE_SET ||--o{ FEATURE_VECTOR : defines
    TRAINING_RUN ||--o{ FEATURE_VECTOR : generates
    TRAINING_RUN ||--o{ MODEL_VERSION : produces
    MODEL_VERSION ||--o{ PREDICTION : generates
    STATION ||--o{ PREDICTION : receives
    OBSERVATION o|--o{ PREDICTION : evaluates
    TRAINING_RUN ||--o{ METRIC : reports
    MODEL_VERSION ||--o{ METRIC : scores
    STATION o|--o{ METRIC : groups
    PIPELINE_EXECUTION ||--o{ TRAINING_RUN : starts
    PIPELINE_EXECUTION ||--o{ PREDICTION : publishes
    PIPELINE_EXECUTION ||--o{ DRIFT_SIGNAL : detects

    STATION {
        string station_id PK
        string station_name
        string corridor
        decimal latitude
        decimal longitude
    }

    OBSERVATION {
        string station_id PK, FK
        timestamp observed_at PK
        integer demand "target"
        string dataset_version FK
    }

    CONTEXT {
        timestamp observed_at PK
        decimal rain_mm
        decimal rain_forecast
        decimal temperature_c
        decimal temperature_forecast
        decimal event_intensity
        string dataset_version FK
    }

    FEATURE_SET {
        string feature_set_id PK
        string version
        string definition_hash
        timestamp created_at
    }

    FEATURE_VECTOR {
        string feature_set_id PK, FK
        string station_id PK, FK
        timestamp observed_at PK
        string training_run_id FK
        decimal lag_1
        decimal lag_4
        decimal lag_96
        decimal rolling_mean_96
        integer local_hour
        integer local_weekday
    }

    TRAINING_RUN {
        string training_run_id PK
        string execution_id FK
        string dataset_version FK
        string feature_set_id FK
        timestamp train_start
        timestamp train_end
        timestamp validation_start
        timestamp validation_end
        timestamp cutoff_at
        string code_commit
        string status
    }

    MODEL_VERSION {
        string model_version_id PK
        string training_run_id FK
        string algorithm
        string hyperparameters_json
        string artifact_uri
        timestamp trained_at
        boolean is_active
    }

    PREDICTION {
        string prediction_id PK
        string model_version_id FK
        string execution_id FK
        string station_id FK
        timestamp target_at
        timestamp created_at
        integer horizon_steps
        decimal predicted_demand
        integer actual_demand FK
        decimal absolute_error
        string submission_status
    }

    METRIC {
        string metric_id PK
        string training_run_id FK
        string model_version_id FK
        string station_id FK
        string metric_name
        timestamp window_start
        timestamp window_end
        decimal metric_value
        timestamp calculated_at
    }

    PIPELINE_EXECUTION {
        string execution_id PK
        timestamp started_at
        timestamp finished_at
        timestamp last_observed_at
        string cursor
        string status
        string error_message
    }

    DRIFT_SIGNAL {
        string drift_signal_id PK
        string execution_id FK
        string feature_name
        string drift_type
        timestamp window_start
        timestamp window_end
        decimal score
        decimal threshold
        boolean triggered_retraining
    }
```

## Decisiones de diseño

### Grano de las tablas fuente

- `OBSERVATION` tiene una fila por estación y timestamp. Su clave primaria es
  `(station_id, observed_at)`; `demand` es el objetivo que se quiere predecir.
- `CONTEXT` tiene una fila por timestamp y se relaciona con todas las estaciones
  mediante `observed_at`. Esto evita duplicar clima y eventos 12 veces.
- `STATION` conserva los ceros iniciales del identificador y las coordenadas
  para features geográficas y visualizaciones.

### Reproducibilidad del entrenamiento

- `FEATURE_SET` versiona la definición y el hash de las variables.
- `FEATURE_VECTOR` conserva el valor calculado de cada feature en el grano
  estación-timestamp. Así se pueden reconstruir predicciones y auditar fugas de
  información.
- `TRAINING_RUN` registra el cutoff y las ventanas temporales. La validación
  debe ocurrir después de `train_end` y antes de `validation_end`.
- `MODEL_VERSION` apunta al artefacto exacto y al commit que lo produjo. Solo
  una versión debería estar activa para cada objetivo y horizonte.

### Predicción y monitoreo

- `PREDICTION` permite almacenar cuatro horizontes usando `horizon_steps` y
  deja `actual_demand` vacío hasta que la observación real esté disponible.
- `METRIC` soporta WAPE, accuracy y métricas rolling, tanto globales como por
  estación y modelo.
- `PIPELINE_EXECUTION` conserva el cursor o último timestamp procesado para
  que la operación incremental sea reanudable e idempotente.
- `DRIFT_SIGNAL` registra data drift y concept drift, el umbral aplicado y si
  la señal provocó reentrenamiento.

## Restricciones recomendadas

1. No permitir features calculadas con timestamps posteriores a `observed_at`.
2. Unicidad en `PREDICTION` para `(model_version_id, station_id, target_at,
   horizon_steps)`.
3. Validar que `actual_demand` corresponda al mismo `station_id` y `target_at`.
4. Mantener inmutables `OBSERVATION`, `CONTEXT`, `FEATURE_VECTOR` y las
   versiones de modelo; las correcciones deben producir una nueva versión.
5. Indexar `observed_at`, `(station_id, observed_at)`, `target_at` y
   `execution_id` para ingesta, backtesting y dashboard.