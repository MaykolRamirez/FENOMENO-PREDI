"""Descarga de series mensuales desde la API de BCRPData.

Uso:
    python src/descarga_bcrp.py                 # series objetivo y predictoras de config.py
    python src/descarga_bcrp.py --desde 1995-1  # cambiar el inicio

Alternativa sin internet: leer un Excel descargado a mano desde BCRPData
con `leer_excel_bcrp("ruta.xlsx")`.

Salidas:
    data/raw/bcrp_mensual.csv              tabla larga: fecha, codigo, valor
    data/vintages/bcrp_mensual_<fecha>.csv copia fechada de la descarga
"""
from __future__ import annotations

import argparse
import re
from datetime import datetime

import pandas as pd
import requests

from config import DATA_RAW, DATA_VINTAGES, PREDICTORAS_BCRP, SERIES_OBJETIVO

URL_API = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api/{codigos}/json/{desde}/{hasta}/esp"

MESES = {
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
    "jul": 7, "ago": 8, "sep": 9, "set": 9, "oct": 10, "nov": 11, "dic": 12,
    # por si la API responde en inglés
    "jan": 1, "apr": 4, "aug": 8, "dec": 12,
}


def a_periodo(etiqueta: str) -> pd.Period:
    """Convierte etiquetas del BCRP a periodo mensual.

    Acepta el formato de la API ("Ene.1995") y el del Excel ("Ene95").

    >>> str(a_periodo("Ene.1995")), str(a_periodo("Jul26")), str(a_periodo("Set.2026"))
    ('1995-01', '2026-07', '2026-09')
    """
    m = re.match(r"([A-Za-z]{3})\.?\s*(\d{2,4})$", etiqueta.strip())
    if not m:
        raise ValueError(f"Fecha no reconocida: {etiqueta!r}")
    mes = MESES[m.group(1).lower()]
    anio = int(m.group(2))
    if anio < 100:
        anio += 1900 if anio >= 50 else 2000
    return pd.Period(year=anio, month=mes, freq="M")


def a_numero(valor) -> float | None:
    """La API devuelve texto; 'n.d.' significa dato no disponible."""
    try:
        return float(str(valor).replace(",", ""))
    except ValueError:
        return None


def parsear_respuesta(json_bcrp: dict, codigos: list[str]) -> pd.DataFrame:
    """Convierte la respuesta JSON de la API en una tabla larga."""
    filas = []
    for periodo in json_bcrp.get("periods", []):
        fecha = a_periodo(periodo["name"])
        for codigo, valor in zip(codigos, periodo["values"]):
            filas.append({"fecha": fecha, "codigo": codigo, "valor": a_numero(valor)})
    return pd.DataFrame(filas)


def descargar(codigos: list[str], desde: str = "1994-1", hasta: str | None = None) -> pd.DataFrame:
    """Descarga las series UNA POR UNA.

    Importante (verificado el 02/11/2026): si se piden varios códigos juntos, la API
    devuelve las columnas en su propio orden (no en el pedido) y sin los códigos,
    así que no se puede saber qué columna es cada serie. Por eso se pide una a la vez.
    """
    hasta = hasta or f"{datetime.now().year}-12"
    partes = []
    for codigo in codigos:
        url = URL_API.format(codigos=codigo, desde=desde, hasta=hasta)
        try:
            respuesta = requests.get(url, timeout=60)
            respuesta.raise_for_status()
            datos = respuesta.json()
        except Exception as error:  # una serie caída no detiene las demás
            print(f"  {codigo}: NO SE PUDO DESCARGAR ({error})")
            continue
        tabla = parsear_respuesta(datos, [codigo]).dropna(subset=["valor"])
        nombre = datos.get("config", {}).get("series", [{}])[0].get("name", "")
        partes.append(tabla)
        print(f"  {codigo}: {tabla['fecha'].min()} a {tabla['fecha'].max()} · {nombre}")
    return pd.concat(partes, ignore_index=True)


def leer_excel_bcrp(ruta: str) -> pd.DataFrame:
    """Lee un Excel exportado desde la web de BCRPData (hoja 'Mensuales').

    Fila 1: códigos de serie; fila 2: nombres; desde la fila 3: fecha y valores.
    """
    hoja = pd.read_excel(ruta, sheet_name="Mensuales", header=None)
    codigos = hoja.iloc[0, 1:].tolist()
    datos = hoja.iloc[2:].copy()
    datos.columns = ["etiqueta"] + codigos
    datos["fecha"] = datos["etiqueta"].astype(str).map(a_periodo)
    tabla = datos.drop(columns="etiqueta").melt(id_vars="fecha", var_name="codigo", value_name="valor")
    tabla["valor"] = tabla["valor"].map(a_numero)
    return tabla


def guardar(tabla: pd.DataFrame, nombre: str = "bcrp_mensual") -> None:
    """Guarda la tabla y una copia fechada (vintage) que nunca se sobrescribe."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_VINTAGES.mkdir(parents=True, exist_ok=True)
    tabla = tabla.copy()
    tabla["fecha"] = tabla["fecha"].astype(str)
    tabla.to_csv(DATA_RAW / f"{nombre}.csv", index=False)
    sello = datetime.now().strftime("%Y%m%d_%H%M")
    tabla.to_csv(DATA_VINTAGES / f"{nombre}_{sello}.csv.gz", index=False)
    print(f"Guardado: data/raw/{nombre}.csv y vintage {sello}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descarga series de BCRPData")
    parser.add_argument("--desde", default="1994-1")
    parser.add_argument("--hasta", default=None)
    args = parser.parse_args()

    codigos = list(SERIES_OBJETIVO) + list(PREDICTORAS_BCRP)
    print(f"Descargando {len(codigos)} series desde BCRPData…")
    tabla = descargar(codigos, args.desde, args.hasta)
    guardar(tabla)
    print(tabla.groupby("codigo")["fecha"].agg(["min", "max", "count"]))
