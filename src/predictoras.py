"""Tabla mensual de predictoras y reporte de disponibilidad (tarea de R1).

Uso:  python src/predictoras.py   (después de descarga_bcrp.py y temperatura_mar.py)

Qué hace:
1. Toma las predictoras de config.PREDICTORAS_BCRP desde data/raw/bcrp_mensual.csv.
2. Las series en nivel (precios de metales, IGV) se convierten a variación % interanual,
   igual que las series objetivo. Las que ya vienen en variación % se dejan como están.
3. Regla 2 de la propuesta: el IGV interno recaudado en el mes m corresponde a la actividad
   del mes m-1, por eso se asigna al mes anterior.
4. Agrega la anomalía Niño 1+2 (NOAA) y su cambio en 3 meses, y la electricidad del COES
   (var. % 12 meses; el mes en curso se compara a igual número de días).

Salidas:
    data/processed/predictoras_mensual.csv       una columna por predictora (variación %)
    resultados/disponibilidad_predictoras.csv     último mes disponible de cada fuente y cuántos
                                                  meses se adelanta a la cifra oficial
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from config import DATA_PROCESSED, DATA_RAW, PREDICTORAS_BCRP, RESULTADOS, SERIES_OBJETIVO

EN_NIVEL = {"PN01652XM", "PN01654XM", "PN01657XM", "PN38697FM"}  # se pasan a var. % 12 meses
MES_ANTERIOR = {"PN38697FM"}  # IGV interno: recaudación de m = actividad de m-1


def construir() -> tuple[pd.DataFrame, pd.DataFrame]:
    bcrp = pd.read_csv(DATA_RAW / "bcrp_mensual.csv")
    bcrp["fecha"] = pd.PeriodIndex(bcrp["fecha"], freq="M")
    ancho = bcrp.pivot_table(index="fecha", columns="codigo", values="valor").sort_index()
    ancho = ancho.reindex(pd.period_range(ancho.index.min(), ancho.index.max(), freq="M"))

    tabla = pd.DataFrame(index=ancho.index)
    for codigo in PREDICTORAS_BCRP:
        if codigo not in ancho:
            print(f"  {codigo}: no está en la descarga, se omite")
            continue
        serie = ancho[codigo]
        if codigo in EN_NIVEL:
            serie = 100 * (serie / serie.shift(12) - 1)
        if codigo in MES_ANTERIOR:
            serie = serie.shift(-1)
        tabla[codigo] = serie.round(3)

    nino = DATA_RAW / "nino12_mensual.csv"
    if nino.exists():
        n = pd.read_csv(nino)
        n.index = pd.PeriodIndex(n["fecha"], freq="M")
        tabla = tabla.reindex(tabla.index.union(n.index))
        tabla["nino12"] = n["anomalia_nino12"]
        tabla["nino12_cambio3m"] = n["cambio_3m"]

    coes = DATA_RAW / "coes_mensual.csv"
    if coes.exists():
        c = pd.read_csv(coes)
        c.index = pd.PeriodIndex(c["fecha"], freq="M")
        tabla = tabla.reindex(tabla.index.union(c.index))
        for col in ("var12_scada", "var12_medidores"):
            tabla[f"coes_{col}"] = c[col]

    ultimo_oficial = max(ancho[c].dropna().index.max() for c in SERIES_OBJETIVO if c in ancho)
    filas = []
    for col in tabla.columns:
        ultimo = tabla[col].dropna().index.max()
        filas.append({"predictora": col,
                      "descripcion": PREDICTORAS_BCRP.get(col, "Electricidad SEIN, var. % 12 meses (COES)" if col.startswith("coes")
                                                      else "Temperatura del mar (NOAA)"),
                      "ultimo_mes": str(ultimo),
                      "meses_de_adelanto": (ultimo - ultimo_oficial).n,
                      "revisado": date.today().isoformat()})
    disponibilidad = pd.DataFrame(filas)
    print(f"Última cifra oficial: {ultimo_oficial}")
    print(disponibilidad[["predictora", "ultimo_mes", "meses_de_adelanto"]].to_string(index=False))
    return tabla, disponibilidad


if __name__ == "__main__":
    tabla, disponibilidad = construir()
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    salida = tabla.copy()
    salida.index = salida.index.astype(str)
    salida.to_csv(DATA_PROCESSED / "predictoras_mensual.csv", index_label="fecha")
    disponibilidad.to_csv(RESULTADOS / "disponibilidad_predictoras.csv", index=False)
    print("Guardado: data/processed/predictoras_mensual.csv y resultados/disponibilidad_predictoras.csv")
