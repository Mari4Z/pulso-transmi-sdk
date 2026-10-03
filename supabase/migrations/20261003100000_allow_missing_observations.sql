-- Fase final (contrato de observación v2): una observación puede llegar con
-- quality = "missing" y sin valor. Un faltante no es cero, así que la columna
-- debe aceptar NULL en vez de forzar un entero; el check >= 0 sigue aplicando
-- a los valores reales.
alter table public.observations alter column demand drop not null;
