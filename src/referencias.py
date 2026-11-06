"""Modelos de referencia y evaluación con origen móvil (parte 1 del proyecto).

Tres referencias por serie objetivo (variación % interanual):
    1. Persistencia:  ŷ_t = y_{t-k}
    2. AR:            y_t = a + b1·y_{t-k} + b2·y_{t-k-1} + b3·y_{t-12} + e
    3. ARX:           AR + electricidad_{t-k} + Niño1+2_t + cambio 3 meses de Niño1+2_t

k = rezago de información (meses entre el mes estimado y la última cifra publicada):
    k = 1  estimación hecha entre el día 1 y el 14 del mes t+2 (como hoy, 02/11, para setiembre)
    k = 2  estimación hecha entre el día 15 del mes t+1 y el 14 del mes t+2 (punto de operación
           de la propuesta, unos 30 días antes de la cifra oficial)
La electricidad del BCRP sale casi junto con la cifra del INEI, por eso se usa con el mismo
rezago k. La temperatura del mar (NOAA, semanal) sí está disponible para el mes t.

Evaluación con origen móvil: para cada mes t desde 2010-01, los modelos se estiman SOLO con
meses cuya cifra ya estaba publicada (≤ t-k) y luego se pronostica t. Los meses de la pandemia
(2020-03 a 2021-12) se excluyen del entrenamiento y se reportan aparte en las métricas.

4. REG (selección por régimen, desde la v2): cada mes t se clasifica como "El Niño" si la
   anomalía Niño 1+2 del mes supera UMBRAL_NINO (°C). En cada serie se usa la referencia con
   menor error pasado EN EL MISMO RÉGIMEN, calculado solo con meses ya publicados (≤ t-k).
   Motivo: en Pesca el AR es mejor en promedio, pero la persistencia gana en los tres episodios
   de El Niño (hallazgo de la parte 1). La regla es automática; no se elige a mano por sector.

Intervalo: percentiles 10 y 90 de los últimos 60 errores fuera de muestra ya observados en el
mismo régimen (si hay menos de 24, se usan todos). Versión simple de calibración conforme
condicionada al régimen; la calibración completa es tarea de la parte 2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATA_PROCESSED, DATA_RAW, DATA_VINTAGES, SERIES_OBJETIVO

COVID = pd.period_range("2020-03", "2021-12", freq="M")
CODIGO_ELECTRICIDAD = "PD37965AM"
MODELOS_BASE = ("persistencia", "AR", "ARX")
MODELOS = MODELOS_BASE + ("REG",)
UMBRAL_NINO = 1.0  # °C de anomalía Niño 1+2 para considerar el mes "El Niño"


def _leer(nombre: str) -> pd.DataFrame:
    """Lee data/raw/<nombre>.csv; si no existe (repositorio recién clonado), usa el vintage
    más reciente de data/vintages/ (.csv o .csv.gz)."""
    ruta = DATA_RAW / f"{nombre}.csv"
    if ruta.exists():
        return pd.read_csv(ruta)
    candidatos = sorted(DATA_VINTAGES.glob(f"{nombre}_*.csv*"))
    if not candidatos:
        raise FileNotFoundError(f"No hay {nombre}: ejecute la descarga (src/descarga_bcrp.py, src/temperatura_mar.py).")
    print(f"Usando vintage {candidatos[-1].name}")
    return pd.read_csv(candidatos[-1])


def cargar() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (Y, X): Y con una columna por serie objetivo; X con predictoras exógenas."""
    bcrp = _leer("bcrp_mensual")
    bcrp["fecha"] = pd.PeriodIndex(bcrp["fecha"], freq="M")
    ancho = bcrp.pivot(index="fecha", columns="codigo", values="valor").sort_index()
    nino = _leer("nino12_mensual")
    nino["fecha"] = pd.PeriodIndex(nino["fecha"], freq="M")
    nino = nino.set_index("fecha")
    indice = pd.period_range(ancho.index.min(), nino.index.max(), freq="M")
    Y = ancho.reindex(indice)[list(SERIES_OBJETIVO)]
    X = pd.DataFrame(index=indice)
    X["electricidad"] = ancho.reindex(indice)[CODIGO_ELECTRICIDAD]
    X["nino12"] = nino["anomalia_nino12"].reindex(indice)
    X["nino12_cambio3m"] = nino["cambio_3m"].reindex(indice)
    return Y, X


def tabla_rezagos(y: pd.Series, X: pd.DataFrame, k: int) -> pd.DataFrame:
    """Tabla de modelado de UNA serie respetando el calendario de publicación."""
    t = pd.DataFrame(index=y.index)
    t["y"] = y
    t[f"y_lag{k}"] = y.shift(k)
    t[f"y_lag{k + 1}"] = y.shift(k + 1)
    t["y_lag12"] = y.shift(12)
    t[f"elec_lag{k}"] = X["electricidad"].shift(k)
    t["nino12"] = X["nino12"]
    t["nino12_cambio3m"] = X["nino12_cambio3m"]
    return t


def _ols(Xm: np.ndarray, ym: np.ndarray) -> np.ndarray:
    Xc = np.column_stack([np.ones(len(Xm)), Xm])
    beta, *_ = np.linalg.lstsq(Xc, ym, rcond=None)
    return beta


def _predecir(beta: np.ndarray, fila: np.ndarray) -> float:
    return float(beta[0] + fila @ beta[1:])


def evaluar_serie(y: pd.Series, X: pd.DataFrame, k: int, inicio: str = "2010-01",
                  fin: str | None = None, minimo_train: int = 60) -> pd.DataFrame:
    """Pronósticos fuera de muestra (origen móvil) de las tres referencias para una serie."""
    tab = tabla_rezagos(y, X, k)
    cols_ar = [f"y_lag{k}", f"y_lag{k + 1}", "y_lag12"]
    cols_arx = cols_ar + [f"elec_lag{k}", "nino12", "nino12_cambio3m"]
    fin = pd.Period(fin, "M") if fin else y.dropna().index.max()
    filas = []
    for t in pd.period_range(inicio, fin, freq="M"):
        ultimo_publicado = t - k
        train = tab.loc[:ultimo_publicado]
        train = train[~train.index.isin(COVID)]
        fila = {"fecha": t, "real": tab.at[t, "y"] if t in tab.index else np.nan,
                "persistencia": tab.at[t, f"y_lag{k}"]}
        for nombre, cols in (("AR", cols_ar), ("ARX", cols_arx)):
            d = train[["y"] + cols].dropna()
            x_t = tab.loc[t, cols]
            if len(d) < minimo_train or x_t.isna().any():
                fila[nombre] = np.nan
                continue
            beta = _ols(d[cols].to_numpy(), d["y"].to_numpy())
            fila[nombre] = _predecir(beta, x_t.to_numpy(dtype=float))
        filas.append(fila)
    return pd.DataFrame(filas).set_index("fecha")


def agregar_regimen(pron: pd.DataFrame, nino: pd.Series, k: int, minimo: int = 6) -> pd.DataFrame:
    """Añade la columna 'nino' (régimen del mes) y el modelo REG (selección por régimen)."""
    salida = pron.copy()
    salida["nino"] = (nino.reindex(salida.index) > UMBRAL_NINO)
    abs_err = {m: (salida["real"] - salida[m]).abs() for m in MODELOS_BASE}
    valido = ~salida.index.isin(COVID)
    reg, elegido = [], []
    for t in salida.index:
        pasado = valido & (salida.index <= t - k) & salida["real"].notna()
        mismo = pasado & (salida["nino"] == salida.at[t, "nino"])
        filtro = mismo if mismo.sum() >= minimo else pasado
        mae = {m: abs_err[m][filtro].mean() for m in MODELOS_BASE}
        mae = {m: v for m, v in mae.items() if pd.notna(v) and pd.notna(salida.at[t, m])}
        mejor = min(mae, key=mae.get) if mae else "persistencia"
        elegido.append(mejor)
        reg.append(salida.at[t, mejor])
    salida["REG"] = reg
    salida["REG_modelo"] = elegido
    return salida


def agregar_intervalos(pron: pd.DataFrame, k: int, ventana: int = 60, minimo: int = 24) -> pd.DataFrame:
    """Añade p10 y p90 por modelo con los errores fuera de muestra ya observados en el mismo
    régimen (columna 'nino'); si no hay suficientes, usa todos los errores pasados."""
    salida = pron.copy()
    regimen = salida["nino"] if "nino" in salida else pd.Series(False, index=salida.index)
    for m in [m for m in MODELOS if m in salida]:
        err = (salida["real"] - salida[m])
        err = err[~err.index.isin(COVID)].dropna()
        p10, p90 = [], []
        for t in salida.index:
            pasado = err.loc[: t - k]
            mismo = pasado[regimen.reindex(pasado.index) == regimen.at[t]].tail(ventana)
            usar = mismo if len(mismo) >= minimo else pasado.tail(ventana)
            if len(usar) < minimo:
                p10.append(np.nan); p90.append(np.nan); continue
            p10.append(salida.at[t, m] + usar.quantile(0.10))
            p90.append(salida.at[t, m] + usar.quantile(0.90))
        salida[f"{m}_p10"] = p10
        salida[f"{m}_p90"] = p90
    return salida


def metricas(pron: pd.DataFrame, desde: str = "2010-01", excluir_covid: bool = True,
             regimen: str = "todos") -> pd.DataFrame:
    """MAE, RMSE y cobertura del intervalo 80 %. regimen: 'todos', 'nino' o 'normal'."""
    d = pron.loc[pd.Period(desde, "M"):].dropna(subset=["real"])
    if excluir_covid:
        d = d[~d.index.isin(COVID)]
    if regimen != "todos" and "nino" in d:
        d = d[d["nino"] == (regimen == "nino")]
    filas = {}
    for m in [m for m in MODELOS if m in d]:
        e = (d["real"] - d[m]).dropna()
        cubre = ((d["real"] >= d[f"{m}_p10"]) & (d["real"] <= d[f"{m}_p90"]))[d[f"{m}_p10"].notna()] \
            if f"{m}_p10" in d else pd.Series(dtype=float)
        filas[m] = {"MAE": e.abs().mean(), "RMSE": np.sqrt((e ** 2).mean()),
                    "cobertura_80": cubre.mean() if len(cubre) else np.nan, "n": len(e)}
    t = pd.DataFrame(filas).T
    t["MAE_relativo"] = t["MAE"] / t.loc["persistencia", "MAE"]
    return t


def ejecutar(k: int, inicio: str = "2010-01") -> dict[str, pd.DataFrame]:
    """Evalúa las tres referencias para las diez series. Devuelve {codigo: pronósticos}."""
    Y, X = cargar()
    salida = {}
    for c in SERIES_OBJETIVO:
        pron = evaluar_serie(Y[c], X, k, inicio, fin=str(Y[c].dropna().index.max() + k))
        salida[c] = agregar_intervalos(agregar_regimen(pron, X["nino12"], k), k)
    return salida


def guardar_tabla_procesada(k: int = 2) -> None:
    """Guarda la tabla de modelado en formato largo (una fila por mes y serie)."""
    Y, X = cargar()
    partes = []
    for c, nombre in SERIES_OBJETIVO.items():
        t = tabla_rezagos(Y[c], X, k).rename(columns=lambda s: s.replace(f"lag{k}", "lag_k")
                                             .replace(f"lag{k + 1}", "lag_k1"))
        t.insert(0, "serie", nombre); t.insert(0, "codigo", c)
        partes.append(t)
    tabla = pd.concat(partes).reset_index(names="fecha")
    tabla["fecha"] = tabla["fecha"].astype(str)
    tabla["k"] = k
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(DATA_PROCESSED / f"tabla_modelado_k{k}.csv", index=False)
