"""Parte 1: evalúa las referencias y genera tablas y figuras en resultados/.

Uso:  python src/parte1.py        (tarda ~1 minuto)
No registra estimaciones: eso se hace a mano con registro.registrar() y un commit.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import referencias as R
from config import EPISODIOS_NINO, RESULTADOS, SERIES_OBJETIVO

AZUL, NARANJA, GRIS = "#1F5FA8", "#C8632B", "#6b6b6b"
EPISODIOS = {"El Niño 2017": ("2017-01", "2017-05"), "El Niño 2023-24": ("2023-03", "2024-04"),
             "El Niño 2026": ("2026-03", "2026-08")}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "axes.titleweight": "bold"})


def tablas(res: dict[int, dict[str, pd.DataFrame]]) -> None:
    filas, nino = [], []
    for k, P in res.items():
        for c, n in SERIES_OBJETIVO.items():
            for desde, cov in (("2010-01", True), ("2015-01", True), ("2010-01", False)):
                for reg in ("todos", "nino", "normal"):
                    m = R.metricas(P[c], desde=desde, excluir_covid=cov, regimen=reg).round(3)
                    periodo = f"{desde[:4]}-2026 " + ("sin pandemia" if cov else "con pandemia")
                    for mod, r in m.iterrows():
                        filas.append({"k": k, "codigo": c, "serie": n, "periodo": periodo,
                                      "regimen": reg, "modelo": mod, **r})
            for ep, (a, b) in EPISODIOS.items():
                s = P[c].loc[pd.Period(a, "M"):pd.Period(b, "M")]
                for mod in R.MODELOS:
                    nino.append({"k": k, "serie": n, "episodio": ep, "modelo": mod,
                                 "MAE": round((s["real"] - s[mod]).abs().mean(), 3)})
    pd.DataFrame(filas).to_csv(RESULTADOS / "errores_referencias.csv", index=False)
    pd.DataFrame(nino).to_csv(RESULTADOS / "errores_el_nino.csv", index=False)
    pd.concat(res[1], names=["codigo", "fecha"]).round(3).to_csv(
        RESULTADOS / "pronosticos_fuera_de_muestra_k1.csv")


def _sombrear(ax):
    for e in EPISODIOS_NINO.values():
        fin = e["fin"] or pd.Timestamp.today().strftime("%Y-%m")
        ax.axvspan(pd.Period(e["inicio"], "M").to_timestamp(), (pd.Period(fin, "M") + 1).to_timestamp(),
                   color=NARANJA, alpha=0.12, lw=0)


def figuras(res: dict[int, dict[str, pd.DataFrame]]) -> None:
    Y, X = R.cargar()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    y = Y["PN01728AM"].loc["2010-01":].dropna()
    a1.plot(y.index.to_timestamp(), y, color=AZUL, lw=1.6); a1.set_ylim(-12, 16)
    a1.set_title("Producción nacional: variación % interanual")
    n = X["nino12"].loc["2010-01":].dropna()
    a2.plot(n.index.to_timestamp(), n, color=NARANJA, lw=1.6)
    a2.set_title("Anomalía de temperatura del mar, región Niño 1+2 (°C)")
    for a in (a1, a2):
        a.axhline(0, color=GRIS, lw=0.8); _sombrear(a)
    fig.text(0.01, 0.005, "Sombreado: El Niño costero (fechas provisionales, ENFEN). Fuentes: BCRPData, NOAA CPC.",
             fontsize=7, color=GRIS)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(RESULTADOS / "pbi_y_temperatura_mar.png", dpi=160)
    plt.close(fig)

    P = res[1]["PN01728AM"].iloc[-37:]
    t = P.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.fill_between(t, P["AR_p10"], P["AR_p90"], color=AZUL, alpha=0.15, lw=0, label="intervalo 80 % (AR)")
    ax.plot(t, P["AR"], color=AZUL, lw=1.6, ls="--", label="estimación AR fuera de muestra")
    ax.plot(t, P["real"], color="#222", lw=1.8, label="cifra publicada")
    ax.axhline(0, color=GRIS, lw=0.8); ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("Producción nacional: estimación con rezago k = 1 frente a la cifra publicada (%)")
    fig.tight_layout(); fig.savefig(RESULTADOS / "nowcast_pbi_v1.png", dpi=160); plt.close(fig)

    filas = []
    for c, nm in SERIES_OBJETIVO.items():
        m = R.metricas(res[2][c])
        filas.append((nm.replace(" y Otros Impuestos", "").replace("Producción nacional (PBI)", "PBI"),
                      m.at["AR", "MAE_relativo"], m.at["ARX", "MAE_relativo"]))
    d = pd.DataFrame(filas, columns=["s", "AR", "ARX"]).iloc[::-1]
    yy, h = np.arange(len(d)), 0.36
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(yy + h / 2, d["AR"], h, color=AZUL, label="AR")
    ax.barh(yy - h / 2, d["ARX"], h, color=NARANJA, label="ARX (+ electricidad y mar)")
    ax.axvline(1, color="#222", lw=1); ax.set_yticks(yy, d["s"]); ax.grid(axis="y", visible=False)
    ax.set_xlabel("MAE relativo a persistencia (menor que 1 = mejor)")
    ax.set_title("Error de las referencias, k = 2, 2010–2026 sin pandemia")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(RESULTADOS / "mae_relativo_referencias.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    RESULTADOS.mkdir(exist_ok=True)
    res = {k: R.ejecutar(k) for k in (1, 2)}
    R.guardar_tabla_procesada(2)
    tablas(res)
    figuras(res)
    print("Listo: tablas y figuras en resultados/, tabla de modelado en data/processed/")
