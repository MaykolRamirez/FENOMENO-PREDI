# data/raw

Última descarga de cada fuente. No se editan a mano: se regeneran con los scripts de `src/` y se actualizan
**cada día** con la descarga automática (`.github/workflows/descarga_datos.yml`).

| Archivo | Fuente | Script |
|---|---|---|
| `bcrp_mensual.csv` | BCRPData: 10 series objetivo y 7 predictoras (formato largo: fecha, codigo, valor) | `descarga_bcrp.py` |
| `nino12_semanal.csv`, `nino12_mensual.csv` | NOAA CPC, anomalía Niño 1+2 | `temperatura_mar.py` |
| `icen.csv` | IGP/ENFEN, Índice Costero El Niño | `temperatura_mar.py` |
| `coes_mensual.csv` | COES, producción del SEIN en MWh (SCADA y medidores) y var. % 12 meses | `coes.py` |
