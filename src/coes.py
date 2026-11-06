"""Electricidad del SEIN (COES): producción mensual en MWh y variación interanual.

Uso:  python src/coes.py            (descarga incremental; la primera vez trae desde 2013)
      python src/coes.py --desde 2010-01

Fuente (identificada el 03/11/2026 leyendo el JavaScript del portal):
    POST https://www.coes.org.pe/Portal/portalinformacion/Generacion
    datos: fechaInicial=dd/mm/aaaa, fechaFinal=dd/mm/aaaa, indicador=0 (SCADA) | 1 (medidores)
    respuesta JSON: ListadoPorEmpresa = [{Emprnomb, Meditotal (MWh)}, ...]
La suma de Meditotal es la producción total del SEIN en el periodo.

Por qué importa: el BCRP publica la electricidad casi junto con la cifra oficial (rezago 0).
El COES la tiene desde el día siguiente, así que da el mes en curso antes que el INEI.

Salidas:
    data/raw/coes_mensual.csv            fecha, mwh_scada, mwh_medidores, dias, completo
    data/vintages/coes_mensual_<fecha>.csv.gz
    resultados/coes_diagnostico.txt      resumen de la última corrida
"""
from __future__ import annotations

import argparse
import time
from datetime import date, datetime, timedelta

import pandas as pd
import requests

from config import DATA_RAW, DATA_VINTAGES, RESULTADOS

URL = "https://www.coes.org.pe/Portal/portalinformacion/Generacion"
CABECERAS = {"User-Agent": "Mozilla/5.0 (proyecto academico IAPN, PEU-CD INEI)",
             "X-Requested-With": "XMLHttpRequest"}
FUENTES = {"mwh_scada": 0, "mwh_medidores": 1}


def total_mwh(inicio: date, fin: date, indicador: int) -> float | None:
    datos = {"fechaInicial": inicio.strftime("%d/%m/%Y"), "fechaFinal": fin.strftime("%d/%m/%Y"),
             "indicador": indicador}
    for intento in range(3):
        try:
            r = requests.post(URL, data=datos, headers=CABECERAS, timeout=90)
            r.raise_for_status()
            lista = r.json().get("ListadoPorEmpresa") or []
            total = sum(float(x.get("Meditotal") or 0) for x in lista)
            return total if total > 0 else None
        except Exception:
            time.sleep(3 * (intento + 1))
    return None


def mes(periodo: pd.Period, hoy: date) -> dict:
    inicio = periodo.start_time.date()
    fin = min(periodo.end_time.date(), hoy - timedelta(days=1))
    fila = {"fecha": str(periodo), "dias": (fin - inicio).days + 1,
            "completo": fin == periodo.end_time.date()}
    for col, ind in FUENTES.items():
        fila[col] = total_mwh(inicio, fin, ind)
        time.sleep(0.5)
    return fila


def descargar(desde: str = "2013-01") -> pd.DataFrame:
    hoy = date.today()
    archivo = DATA_RAW / "coes_mensual.csv"
    previo = pd.read_csv(archivo) if archivo.exists() else pd.DataFrame(columns=["fecha"])
    meses = pd.period_range(desde, pd.Period(hoy, "M"), freq="M")
    # Se vuelven a pedir los 3 últimos meses (el mes en curso y las revisiones de medidores).
    recientes = set(str(p) for p in meses[-3:])
    ya = set(previo["fecha"].astype(str)) - recientes
    filas = []
    for p in meses:
        if str(p) in ya:
            continue
        filas.append(mes(p, hoy))
        print(f"  {p}: SCADA {filas[-1]['mwh_scada']} · medidores {filas[-1]['mwh_medidores']}")
    nuevo = pd.DataFrame(filas)
    tabla = pd.concat([previo[previo["fecha"].astype(str).isin(ya)], nuevo], ignore_index=True)
    return tabla.sort_values("fecha").reset_index(drop=True)


def variacion_interanual(tabla: pd.DataFrame) -> pd.DataFrame:
    """Var. % 12 meses. Si el último mes está en curso, se compara con el mismo mes del año
    anterior reescalado a los mismos días (aproximación por promedio diario)."""
    t = tabla.copy()
    t.index = pd.PeriodIndex(t["fecha"], freq="M")
    t = t.reindex(pd.period_range(t.index.min(), t.index.max(), freq="M"))
    t["fecha"] = t.index.astype(str)
    for col in FUENTES:
        base = t[col].shift(12)
        ultimo = t.index[-1]
        if not bool(t["completo"].iloc[-1]):
            base.iloc[-1] = base.iloc[-1] * t["dias"].iloc[-1] / (ultimo - 12).days_in_month
        t[col.replace("mwh", "var12")] = (100 * (t[col] / base - 1)).round(3)
    return t.reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", default="2013-01")
    args = ap.parse_args()
    DATA_RAW.mkdir(parents=True, exist_ok=True); DATA_VINTAGES.mkdir(parents=True, exist_ok=True)
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    print("Descargando producción del SEIN (COES)…")
    tabla = variacion_interanual(descargar(args.desde))
    tabla.to_csv(DATA_RAW / "coes_mensual.csv", index=False)
    sello = datetime.now().strftime("%Y%m%d_%H%M")
    tabla.to_csv(DATA_VINTAGES / f"coes_mensual_{sello}.csv.gz", index=False)
    ultimos = tabla.tail(4)[["fecha", "dias", "completo", "mwh_scada", "var12_scada", "mwh_medidores", "var12_medidores"]]
    resumen = f"COES · {datetime.now():%Y-%m-%d %H:%M}\n{len(tabla)} meses\n{ultimos.to_string(index=False)}"
    (RESULTADOS / "coes_diagnostico.txt").write_text(resumen, encoding="utf-8")
    print(resumen)
