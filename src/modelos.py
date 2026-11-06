"""Parte 2 (R2): LightGBM global con cuantiles y evaluación con origen móvil.

Uso:  python src/modelos.py            (k = 2, evaluación desde 2015-01; tarda unos minutos)
      python src/modelos.py --k 1

Modelo global: una sola tabla con las 10 series apiladas (una fila por mes y serie) y la
serie como variable categórica. Se estiman tres modelos de cuantiles (p10, p50, p90).

Información disponible en cada fecha (sin fuga de información):
    rezagos propios          y_{t-k}, y_{t-k-1}, y_{t-12}            (calendario del INEI)
    COES (electricidad SEIN) mes t                                    (adelanta 2 meses)
    temperatura del mar      mes t y cambio en 3 meses                (NOAA, semanal)
    metales (cobre/oro/zinc) mes t, var. % 12 meses                   (promedio del mes)
    BCRP electricidad, cemento, IGV interno: rezago k                (salen junto con la cifra oficial)

Evaluación: para cada mes t desde INICIO, se entrena solo con meses ya publicados (≤ t-k),
sin pandemia (2020-03 a 2021-12), y se pronostica t. Se compara con la mejor referencia de cada
serie en el mismo periodo (resultados/errores_referencias.csv debe existir: correr parte1.py).

Salidas:
    resultados/lightgbm_pronosticos_k{k}.csv
    resultados/lightgbm_vs_referencias_k{k}.csv
"""
from __future__ import annotations

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

import referencias as R
from config import DATA_PROCESSED, RESULTADOS, SERIES_OBJETIVO

CUANTILES = {"p10": 0.10, "p50": 0.50, "p90": 0.90}
MISMO_MES = ["coes_var12_medidores", "nino12", "nino12_cambio3m", "PN01652XM", "PN01654XM", "PN01657XM"]
CON_REZAGO = ["PD37965AM", "PD37967GM", "PN38697FM"]
PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def tabla_global(k: int) -> pd.DataFrame:
    Y, _ = R.cargar()
    pred = pd.read_csv(DATA_PROCESSED / "predictoras_mensual.csv")
    pred.index = pd.PeriodIndex(pred["fecha"], freq="M")
    pred = pred.drop(columns="fecha").reindex(Y.index)
    exog = pd.DataFrame(index=Y.index)
    for c in MISMO_MES:
        exog[c] = pred[c]
    for c in CON_REZAGO:
        exog[f"{c}_lag{k}"] = pred[c].shift(k)
    partes = []
    for i, c in enumerate(SERIES_OBJETIVO):
        y = Y[c]
        t = pd.DataFrame({"y": y, "y_lagk": y.shift(k), "y_lagk1": y.shift(k + 1), "y_lag12": y.shift(12)})
        t["dif_lagk_lag12"] = t["y_lagk"] - t["y_lag12"]
        t = t.join(exog)
        t["serie"] = i
        t["codigo"] = c
        t["mes"] = t.index.month
        partes.append(t)
    return pd.concat(partes)


def evaluar(k: int = 2, inicio: str = "2015-01", residuo: bool = True) -> pd.DataFrame:
    """residuo=True: el modelo predice y - y_{t-k} y se le suma y_{t-k} (parte de la persistencia)."""
    tab = tabla_global(k)
    X_cols = [c for c in tab.columns if c not in ("y", "codigo")]
    fin = tab["y"].dropna().index.max() + k
    filas = []
    for t in pd.period_range(inicio, fin, freq="M"):
        train = tab[(tab.index <= t - k) & ~tab.index.isin(R.COVID) & tab["y"].notna()]
        test = tab[tab.index == t]
        if test.empty:
            continue
        if residuo:
            train = train[train["y_lagk"].notna()]
            objetivo, base = train["y"] - train["y_lagk"], test["y_lagk"].to_numpy()
        else:
            objetivo, base = train["y"], 0.0
        salida = test[["codigo", "y"]].copy()
        for nombre, q in CUANTILES.items():
            m = lgb.LGBMRegressor(objective="quantile", alpha=q, **PARAMS)
            m.fit(train[X_cols], objetivo, categorical_feature=["serie"])
            salida[nombre] = m.predict(test[X_cols]) + base
        salida["fecha"] = str(t)
        filas.append(salida.reset_index(drop=True))
        if t.month == 1:
            print(f"  {t}")
    pron = pd.concat(filas, ignore_index=True)
    s = pron[["p10", "p50", "p90"]].to_numpy()
    s.sort(axis=1)  # evita cruces de cuantiles
    pron[["p10", "p50", "p90"]] = s
    return pron


def comparar(pron: pd.DataFrame, k: int, desde: str = "2015-01") -> pd.DataFrame:
    ref = pd.read_csv(RESULTADOS / "errores_referencias.csv")
    ref = ref[(ref.k == k) & (ref.periodo == "2015-2026 sin pandemia") & (ref.regimen == "todos")
              & ref.modelo.isin(R.MODELOS_BASE)]
    d = pron[pron["y"].notna()].copy()
    d = d[~pd.PeriodIndex(d["fecha"], freq="M").isin(R.COVID) & (d["fecha"] >= desde)]
    filas = []
    for c, nombre in SERIES_OBJETIVO.items():
        x = d[d.codigo == c]
        r = ref[ref.codigo == c].set_index("modelo")
        mejor = r["MAE"].idxmin()
        mae = (x["y"] - x["p50"]).abs().mean()
        filas.append({"serie": nombre, "MAE_lightgbm": round(mae, 3),
                      "cobertura_80_lightgbm": round(((x.y >= x.p10) & (x.y <= x.p90)).mean(), 3),
                      "mejor_referencia": mejor, "MAE_referencia": round(r.at[mejor, "MAE"], 3),
                      "mejora_%": round(100 * (1 - mae / r.at[mejor, "MAE"]), 1), "n": len(x)})
    return pd.DataFrame(filas)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--inicio", default="2015-01")
    ap.add_argument("--nivel", action="store_true", help="predice el nivel (versión 1) en vez del residuo")
    a = ap.parse_args()
    print(f"LightGBM global, k = {a.k}, evaluación desde {a.inicio}, objetivo: {'nivel' if a.nivel else 'residuo'}…")
    pron = evaluar(a.k, a.inicio, residuo=not a.nivel)
    pron.round(3).to_csv(RESULTADOS / f"lightgbm_pronosticos_k{a.k}.csv", index=False)
    comp = comparar(pron, a.k, a.inicio)
    comp.to_csv(RESULTADOS / f"lightgbm_vs_referencias_k{a.k}.csv", index=False)
    print(comp.to_string(index=False))
