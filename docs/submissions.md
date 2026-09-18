# Envío de predicciones

## Requisitos

- API base: `https://pulso-transmi.72-60-245-2.sslip.io`
- Variable de entorno: `PULSO_API_KEY`
- Header: `Authorization: Bearer $PULSO_API_KEY`
- La API key no debe escribirse en el código ni versionarse.

El contrato oficial está disponible en `/openapi.json`. Antes de enviar,
consulta el ciclo activo:

```bash
curl -sS "$PULSO_API_URL/v1/forecast-cycles/current" \
  -H "Authorization: Bearer $PULSO_API_KEY"
```

El ciclo devuelve `cycle_id`, `data_cutoff`, estado, estaciones esperadas y
timestamps objetivo. Solo se debe enviar cuando `state` sea `open` y deben
respetarse exactamente sus objetivos.

## Formato de submission

La ruta oficial es `POST /v1/submissions`. El body requiere:

```json
{
  "schema_version": "1.0",
  "cycle_id": "cyc_practice_20260918",
  "client_run_id": "run-20260918-hgb-poisson-v1",
  "data_cutoff": "2026-09-09T04:45:00Z",
  "model": {
    "version": "hgb-poisson-v1",
    "trained_at": "2026-09-18T20:13:00Z",
    "training_data_end": "2026-09-02T04:45:00Z",
    "git_commit": "<commit-local>"
  },
  "predictions": [
    {
      "station_id": "02300",
      "target_at": "2026-09-09T05:00:00Z",
      "value": 114.022
    }
  ]
}
```

Debe haber una predicción por cada objetivo del ciclo. Los valores deben ser
numéricos y no negativos. `Idempotency-Key` es obligatorio y debe ser estable
para evitar envíos duplicados:

```bash
curl -sS -X POST "$PULSO_API_URL/v1/submissions" \
  -H "Authorization: Bearer $PULSO_API_KEY" \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: run-20260918-hgb-poisson-v1" \
  --data-binary @submission.json
```

Una respuesta `201` con `status: "accepted"` confirma que el API recibió el
envío. Guarda el `submission_id` y consulta su estado:

```bash
curl -sS "$PULSO_API_URL/v1/submissions/<submission_id>" \
  -H "Authorization: Bearer $PULSO_API_KEY"
```

## Envío realizado

El modelo `hgb-poisson-v1` se envió al ciclo de práctica
`cyc_practice_20260918` con 12 predicciones, una por estación.

- Submission ID: `sub_ac85fb2075d84c9b9a7c4c22662ab669`
- Estado: `accepted`
- HTTP: `201 Created`
- Predicciones recibidas: `12`
- Envío oficial: `true`
- Intento: `1`
- Fecha de recepción: `2026-09-18T20:48:44Z`

El estado se confirmó con `GET /v1/submissions/{submission_id}`. La cantidad
de 12 predicciones se tomó de la respuesta `201` original.