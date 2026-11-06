"""Temperatura del mar: anomalía semanal Niño 1+2 (NOAA) e ICEN (IGP/ENFEN).

Uso:
    python src/temperatura_mar.py

Salidas:
    data/raw/nino12_semanal.csv   semana, temperatura y anomalía de la región Niño 1+2
    data/raw/nino12_mensual.csv   promedio mensual de la anomalía y su cambio en 3 meses
    data/raw/icen.csv             Índice Costero El Niño (mensual)

Recordatorio: el ICEN es una media móvil de 3 meses y su último valor es
provisional. En el modelo solo se usa rezagado o para fechar episodios.
"""
from __future__ import annotations

import re
from datetime import datetime

import pandas as pd
import requests

from config import DATA_RAW, DATA_VINTAGES

URL_NOAA_SEMANAL = "https://www.cpc.ncep.noaa.gov/data/indices/wksst9120.for"
URL_ICEN = "http://met.igp.gob.pe/datos/ICEN.txt"


def parsear_noaa_semanal(texto: str) -> pd.DataFrame:
    """Lee el archivo semanal de la NOAA (columnas: semana y pares SST/SSTA por región).

    Las columnas vienen pegadas cuando la anomalía es negativa ("23.4-0.4"),
    por eso se extraen los números con una expresión regular.
    """
    filas = []
    for linea in texto.splitlines():
        m = re.match(r"\s*(\d{2}[A-Z]{3}\d{4})(.*)", linea)
        if not m:
            continue
        numeros = [float(x) for x in re.findall(r"-?\d+\.\d+", m.group(2))]
        if len(numeros) < 2:
            continue
        filas.append({
            "semana": pd.to_datetime(m.group(1), format="%d%b%Y"),
            "sst_nino12": numeros[0],
            "anomalia_nino12": numeros[1],
        })
    return pd.DataFrame(filas)


def a_mensual(semanal: pd.DataFrame) -> pd.DataFrame:
    """Promedio mensual de la anomalía y su cambio respecto de 3 meses antes."""
    mensual = (semanal.assign(fecha=semanal["semana"].dt.to_period("M"))
               .groupby("fecha", as_index=False)["anomalia_nino12"].mean())
    mensual["cambio_3m"] = mensual["anomalia_nino12"].diff(3)
    return mensual


def parsear_icen(texto: str) -> pd.DataFrame:
    """Lee el ICEN: líneas con año, mes y valor (se ignoran encabezados)."""
    filas = []
    for linea in texto.splitlines():
        partes = linea.split()
        if len(partes) >= 3 and partes[0].isdigit() and partes[1].isdigit():
            try:
                valor = float(partes[2])
            except ValueError:
                continue
            filas.append({"fecha": pd.Period(year=int(partes[0]), month=int(partes[1]), freq="M"),
                          "icen": valor})
    return pd.DataFrame(filas)


def descargar_texto(url: str) -> str:
    respuesta = requests.get(url, timeout=60)
    respuesta.raise_for_status()
    return respuesta.text


if __name__ == "__main__":
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_VINTAGES.mkdir(parents=True, exist_ok=True)
    sello = datetime.now().strftime("%Y%m%d_%H%M")

    print("Descargando anomalía semanal Niño 1+2 (NOAA)…")
    semanal = parsear_noaa_semanal(descargar_texto(URL_NOAA_SEMANAL))
    mensual = a_mensual(semanal)
    semanal.to_csv(DATA_RAW / "nino12_semanal.csv", index=False)
    mensual.astype({"fecha": str}).to_csv(DATA_RAW / "nino12_mensual.csv", index=False)
    semanal.to_csv(DATA_VINTAGES / f"nino12_semanal_{sello}.csv.gz", index=False)
    mensual.astype({"fecha": str}).to_csv(DATA_VINTAGES / f"nino12_mensual_{sello}.csv.gz", index=False)
    print(f"  {len(semanal)} semanas, de {semanal['semana'].min():%d/%m/%Y} a {semanal['semana'].max():%d/%m/%Y}")

    print("Descargando ICEN (IGP/ENFEN)…")
    try:
        icen = parsear_icen(descargar_texto(URL_ICEN))
        icen.astype({"fecha": str}).to_csv(DATA_RAW / "icen.csv", index=False)
        icen.astype({"fecha": str}).to_csv(DATA_VINTAGES / f"icen_{sello}.csv.gz", index=False)
        print(f"  {len(icen)} meses, último: {icen['fecha'].max()} (provisional)")
    except Exception as error:  # el ICEN es complementario; la NOAA ya quedó guardada
        print(f"  ICEN no disponible ({error}); se continúa sin él.")
