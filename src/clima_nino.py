"""Monitor de El Niño en el Perú: lluvias de los últimos 15 días, temperatura del mar y riesgo por lluvia intensa.

Uso:
    python src/clima_nino.py              (descarga y arma los JSON del tablero)
    python src/clima_nino.py --sin-red    (solo recalcula con lo ya descargado)

Fuentes (todas públicas y sin clave):
    Open-Meteo forecast   lluvia diaria observada/modelada de los últimos 15 días y pronóstico de 5 días
    Open-Meteo archive    climatología 2010-2025 (ERA5) para saber qué es "normal" en cada punto
    Open-Meteo marine     temperatura superficial del mar frente a la costa
    NOAA CPC / IGP-ENFEN  anomalía Niño 1+2 e ICEN (los baja src/temperatura_mar.py)

Salidas en dashboard/data/:
    lluvia.json     por punto: serie de 15 días, pronóstico, acumulados, anomalía e índice de riesgo
    mar.json        Niño 1+2 semanal (2 años), ICEN, récords y temperatura del mar por puerto
    resumen.json    fecha de corte, titular y cifras clave
    departamentos.geojson   límites departamentales (se baja una vez)
Caché: data/clima/climatologia_precip.json (se calcula una sola vez por punto).
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

from config import DATA_RAW, RAIZ

DASH = RAIZ / "dashboard" / "data"
CACHE = RAIZ / "data" / "clima"
URL_FORECAST = "https://api.open-meteo.com/v1/forecast"
URL_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
URL_MARINE = "https://marine-api.open-meteo.com/v1/marine"
URL_GEOJSON = "https://raw.githubusercontent.com/juaneladio/peru-geojson/master/peru_departamental_simple.geojson"
DIAS = 15            # ventana de observación
DIAS_PRON = 5        # pronóstico
CLIM_INI, CLIM_FIN = "2010-01-01", "2025-12-31"

# (nombre, departamento en el GeoJSON, latitud, longitud, región)
PUNTOS = [
    ("Tumbes", "TUMBES", -3.57, -80.45, "Costa norte"),
    ("Piura", "PIURA", -5.19, -80.63, "Costa norte"),
    ("Chiclayo", "LAMBAYEQUE", -6.77, -79.84, "Costa norte"),
    ("Trujillo", "LA LIBERTAD", -8.11, -79.03, "Costa norte"),
    ("Chimbote", "ANCASH", -9.07, -78.59, "Costa central"),
    ("Lima", "LIMA", -12.05, -77.04, "Costa central"),
    ("Callao", "CALLAO", -12.06, -77.12, "Costa central"),
    ("Ica", "ICA", -14.07, -75.73, "Costa sur"),
    ("Arequipa", "AREQUIPA", -16.41, -71.54, "Sierra sur"),
    ("Moquegua", "MOQUEGUA", -17.19, -70.93, "Costa sur"),
    ("Tacna", "TACNA", -18.01, -70.25, "Costa sur"),
    ("Cajamarca", "CAJAMARCA", -7.16, -78.50, "Sierra norte"),
    ("Chachapoyas", "AMAZONAS", -6.23, -77.87, "Sierra norte"),
    ("Huaraz", "ANCASH", -9.53, -77.53, "Sierra norte"),
    ("Huánuco", "HUANUCO", -9.93, -76.24, "Sierra central"),
    ("Cerro de Pasco", "PASCO", -10.68, -76.26, "Sierra central"),
    ("Huancayo", "JUNIN", -12.07, -75.21, "Sierra central"),
    ("Huancavelica", "HUANCAVELICA", -12.78, -74.97, "Sierra central"),
    ("Ayacucho", "AYACUCHO", -13.16, -74.22, "Sierra sur"),
    ("Abancay", "APURIMAC", -13.64, -72.88, "Sierra sur"),
    ("Cusco", "CUSCO", -13.53, -71.97, "Sierra sur"),
    ("Puno", "PUNO", -15.84, -70.02, "Sierra sur"),
    ("Moyobamba", "SAN MARTIN", -6.03, -76.97, "Selva"),
    ("Pucallpa", "UCAYALI", -8.38, -74.55, "Selva"),
    ("Iquitos", "LORETO", -3.75, -73.25, "Selva"),
    ("Puerto Maldonado", "MADRE DE DIOS", -12.59, -69.19, "Selva"),
]
# Puertos con temperatura del mar (punto un poco mar adentro).
PUERTOS = [
    ("Tumbes", -3.6, -80.9), ("Paita", -5.1, -81.3), ("Pimentel", -6.85, -80.2),
    ("Salaverry", -8.2, -79.3), ("Chimbote", -9.1, -78.9), ("Callao", -12.1, -77.4),
    ("Pisco", -13.8, -76.5), ("Ilo", -17.7, -71.6),
]
NIVELES = [(25, "Bajo"), (50, "Moderado"), (75, "Alto"), (101, "Muy alto")]


def pedir(url: str, params: dict, intentos: int = 4) -> dict:
    for i in range(intentos):
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 200:
            return r.json()
        time.sleep(5 * (i + 1))
    r.raise_for_status()


def climatologia() -> dict:
    """Por punto: p95 y p99 de la lluvia diaria en días con lluvia, y acumulado normal por día del año (15 días)."""
    ruta = CACHE / "climatologia_precip.json"
    cache = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
    CACHE.mkdir(parents=True, exist_ok=True)
    for nombre, _, lat, lon, _ in PUNTOS:
        if nombre in cache:
            continue
        d = pedir(URL_ARCHIVE, dict(latitude=lat, longitude=lon, start_date=CLIM_INI, end_date=CLIM_FIN,
                                    daily="precipitation_sum", timezone="America/Lima"))["daily"]
        s = pd.Series(d["precipitation_sum"], index=pd.to_datetime(d["time"])).fillna(0.0)
        humedos = s[s >= 1.0]
        acum15 = s.rolling(DIAS).sum().dropna()
        por_dia = acum15.groupby(acum15.index.dayofyear).mean()
        cache[nombre] = {
            "p95": round(float(np.percentile(humedos, 95)), 1) if len(humedos) > 20 else 5.0,
            "p99": round(float(np.percentile(humedos, 99)), 1) if len(humedos) > 20 else 10.0,
            "acum15_dia_del_anio": {str(k): round(float(v), 1) for k, v in por_dia.items()},
        }
        print(f"  climatología {nombre}: p99 = {cache[nombre]['p99']} mm")
        time.sleep(1.5)
    ruta.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return cache


def nivel(indice: float) -> str:
    return next(n for tope, n in NIVELES if indice < tope)


def indice_riesgo(serie: list[float], pron: list[float], clim: dict, dia_anio: int) -> dict:
    """Índice 0-100 de riesgo por lluvia intensa (huaicos e inundaciones). Orientativo, no oficial.

    Mezcla tres señales respecto de lo normal en ese punto (se acota cada una a [0, 1]):
      - intensidad reciente: la mayor lluvia diaria de los últimos 3 días / p99 de días con lluvia (peso 0.40)
      - pronóstico:          la mayor lluvia diaria de los próximos 3 días / p99                   (peso 0.35)
      - saturación:          acumulado de 15 días / (2 x acumulado normal de esa fecha, mínimo 20 mm) (peso 0.25)
    El p99 se fija en al menos 5 mm para que una llovizna en el desierto no dispare la alerta.
    """
    p99 = max(clim["p99"], 5.0)
    normal15 = clim["acum15_dia_del_anio"].get(str(dia_anio), 0.0)
    reciente = min(1.0, max(serie[-3:]) / p99)
    futuro = min(1.0, max(pron[:3]) / p99) if pron else 0.0
    satura = min(1.0, sum(serie) / max(2 * normal15, 20.0))
    ind = round(100 * (0.40 * reciente + 0.35 * futuro + 0.25 * satura), 1)
    return {"indice": ind, "nivel": nivel(ind), "reciente": round(reciente, 2), "pronostico": round(futuro, 2),
            "saturacion": round(satura, 2), "normal15": normal15}


def lluvia(clim: dict) -> list[dict]:
    lat = ",".join(str(p[2]) for p in PUNTOS)
    lon = ",".join(str(p[3]) for p in PUNTOS)
    resp = pedir(URL_FORECAST, dict(latitude=lat, longitude=lon, daily="precipitation_sum,temperature_2m_max",
                                    past_days=DIAS - 1, forecast_days=DIAS_PRON + 1, timezone="America/Lima"))
    resp = resp if isinstance(resp, list) else [resp]
    hoy = datetime.now(timezone(timedelta(hours=-5))).date()
    salida = []
    for (nombre, dep, la, lo, region), r in zip(PUNTOS, resp):
        fechas = r["daily"]["time"]
        p = [x or 0.0 for x in r["daily"]["precipitation_sum"]]
        i_hoy = fechas.index(hoy.isoformat()) if hoy.isoformat() in fechas else DIAS - 1
        obs, pron = p[: i_hoy + 1][-DIAS:], p[i_hoy + 1:][:DIAS_PRON]
        f_obs = fechas[: i_hoy + 1][-DIAS:]
        riesgo = indice_riesgo(obs, pron, clim[nombre], pd.Timestamp(hoy).dayofyear)
        acum = round(sum(obs), 1)
        normal = riesgo["normal15"]
        salida.append({
            "punto": nombre, "departamento": dep, "region": region, "lat": la, "lon": lo,
            "altitud_m": round(r.get("elevation", 0)),
            "fechas": f_obs, "lluvia_mm": [round(x, 1) for x in obs],
            "pronostico_fechas": fechas[i_hoy + 1:][:DIAS_PRON], "pronostico_mm": [round(x, 1) for x in pron],
            "acum15_mm": acum, "normal15_mm": normal,
            "anomalia_mm": round(acum - normal, 1),
            "anomalia_pct": round(100 * (acum / normal - 1)) if normal >= 5 else None,
            "max_dia_mm": round(max(obs), 1), "dias_con_lluvia": int(sum(x >= 1 for x in obs)),
            "p99_mm": clim[nombre]["p99"], **riesgo,
        })
    return salida


def mar() -> dict:
    sem = pd.read_csv(DATA_RAW / "nino12_semanal.csv", parse_dates=["semana"])
    icen = pd.read_csv(DATA_RAW / "icen.csv")
    ultimo = sem.iloc[-1]
    maximo = sem.loc[sem["anomalia_nino12"].idxmax()]
    reciente = sem[sem["semana"] >= sem["semana"].max() - pd.Timedelta(days=730)]
    # récords de cada episodio fuerte para comparar
    episodios = {}
    for nombre, ini, fin in [("1982-83", "1982-06-01", "1983-08-31"), ("1997-98", "1997-04-01", "1998-06-30"),
                             ("2017", "2017-01-01", "2017-06-30"), ("2023", "2023-03-01", "2024-02-29")]:
        sub = sem[(sem["semana"] >= ini) & (sem["semana"] <= fin)]
        episodios[nombre] = round(float(sub["anomalia_nino12"].max()), 1)
    # temperatura del mar frente a la costa (diaria, 30 días)
    puertos = []
    for nombre, la, lo in PUERTOS:
        try:
            d = pedir(URL_MARINE, dict(latitude=la, longitude=lo, hourly="sea_surface_temperature", past_days=30,
                                       forecast_days=1, timezone="America/Lima"))["hourly"]
            s = pd.Series(d["sea_surface_temperature"], index=pd.to_datetime(d["time"])).dropna()
            dia = s.resample("D").mean().dropna()
            dia = dia[dia.index.date <= datetime.now(timezone(timedelta(hours=-5))).date()]
            puertos.append({"puerto": nombre, "lat": la, "lon": lo,
                            "fechas": [x.strftime("%Y-%m-%d") for x in dia.index], "sst": [round(float(v), 1) for v in dia],
                            "actual": round(float(dia.iloc[-1]), 1),
                            "cambio_7d": round(float(dia.iloc[-1] - dia.iloc[-8]), 1) if len(dia) > 8 else None})
        except Exception as e:  # un puerto caído no debe romper el tablero
            print(f"  aviso: sin temperatura del mar en {nombre}: {e}")
        time.sleep(1)
    return {
        "nino12": {"semanas": [x.strftime("%Y-%m-%d") for x in reciente["semana"]],
                   "anomalia": reciente["anomalia_nino12"].round(1).tolist(), "sst": reciente["sst_nino12"].round(1).tolist()},
        "ultimo": {"semana": ultimo["semana"].strftime("%Y-%m-%d"), "anomalia": float(ultimo["anomalia_nino12"]),
                   "sst": float(ultimo["sst_nino12"])},
        "maximo_historico": {"semana": maximo["semana"].strftime("%Y-%m-%d"), "anomalia": float(maximo["anomalia_nino12"])},
        "episodios_max": episodios,
        "icen": {"meses": icen["fecha"].tail(24).tolist(), "valor": icen["icen"].tail(24).round(2).tolist()},
        "puertos": puertos,
    }


def resumen(puntos: list[dict], m: dict) -> dict:
    por_nivel = pd.Series([p["nivel"] for p in puntos]).value_counts().to_dict()
    altos = sorted([p for p in puntos if p["indice"] >= 50], key=lambda p: -p["indice"])
    norte = [p for p in puntos if p["region"] == "Costa norte"]
    u = m["ultimo"]
    tit = (f"Niño 1+2 en {u['anomalia']:+.1f} °C (semana del {u['semana']}). "
           + (f"{len(altos)} de {len(puntos)} puntos con riesgo alto o muy alto por lluvia: "
              + ", ".join(p["punto"] for p in altos[:4]) + "." if altos
              else f"Sin puntos con riesgo alto por lluvia en los próximos 3 días ({len(puntos)} vigilados)."))
    return {"corte": datetime.now(timezone(timedelta(hours=-5))).strftime("%Y-%m-%d %H:%M"), "titular": tit,
            "niveles": {n: int(por_nivel.get(n, 0)) for _, n in NIVELES},
            "lluvia_15d_costa_norte_mm": round(float(np.mean([p["acum15_mm"] for p in norte])), 1),
            "normal_15d_costa_norte_mm": round(float(np.mean([p["normal15_mm"] for p in norte])), 1),
            "lluvia_max_dia": max(({"punto": p["punto"], "mm": p["max_dia_mm"]} for p in puntos), key=lambda x: x["mm"])}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sin-red", action="store_true", help="no descarga; solo recalcula el resumen con lluvia.json y mar.json")
    a = ap.parse_args()
    DASH.mkdir(parents=True, exist_ok=True)
    if not a.sin_red:
        geo = DASH / "departamentos.geojson"
        if not geo.exists():
            geo.write_bytes(requests.get(URL_GEOJSON, timeout=60).content)
        print("Climatología (se calcula una vez)…")
        clim = climatologia()
        print("Lluvia de los últimos 15 días…")
        pts = lluvia(clim)
        print("Temperatura del mar…")
        m = mar()
        (DASH / "lluvia.json").write_text(json.dumps(pts, ensure_ascii=False), encoding="utf-8")
        (DASH / "mar.json").write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    else:
        pts = json.loads((DASH / "lluvia.json").read_text(encoding="utf-8"))
        m = json.loads((DASH / "mar.json").read_text(encoding="utf-8"))
    res = resumen(pts, m)
    (DASH / "resumen.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(res["titular"])
