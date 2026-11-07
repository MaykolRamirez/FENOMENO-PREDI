"""Test de realidad del modelo de alerta (Y = emergencia SINPAD en las 4 semanas siguientes).

Pregunta: si el modelo hubiera funcionado de verdad cada lunes desde 2008, con solo lo que se sabía ese día,
¿habría avisado mejor que la climatología ("esta época del año este distrito suele tener emergencias")?

Pasos (docs/3_validacion/23_TEST_DE_REALIDAD.md):
  1. Fuga de información. Dos escenarios con el mismo código y la misma validación:
       actual    las variables tal como están en la base (lluvia y caudal de la misma semana t; lluvia y caudal
                 "normales" calculados con 2001-2020, es decir, con años futuros para predecir 2008-2019)
       realista  normales de lluvia y caudal recalculadas SOLO con los años de entrenamiento y todas las variables
                 de clima, río y mar desfasadas 1 semana (el lunes aún no están completos los datos de la semana
                 que acaba de terminar: NASA POWER y GloFAS llegan con días de retraso)
  2. Años reservados 2017 (Niño Costero), 2019 (año normal) y 2023 (Yaku): cada uno lo predice un modelo entrenado
     solo con los años anteriores; el umbral de alerta (5 % de mayor riesgo) se fija en el entrenamiento.
  3. Referencias: climatología, persistencia (¿hubo emergencia en la semana t?) y "alertar siempre en febrero-abril".
  4. Métricas: PR-AUC, ROC-AUC, Brier, Brier skill score frente a la climatología, calibración, métricas por evento
     (episodios avisados con anticipación, falsas alarmas), por plazo h = 1..4, por departamento y por tipo de año.
     Además se vuelve a validar el modelo por plazo (src/prediccion_semanal.py) sin la variable emergencia_t.
  5. Intervalos de confianza: bootstrap por bloques distrito x año y, más exigente, por años completos.

Salidas: docs/evidencia/evaluacion_realista.json, docs/3_validacion/img/realidad_*.png,
         data/modelo/resultados_realista/ (métricas y predicciones)
Uso: python herramientas/pruebas/evaluacion_realista.py [--rapido]   (--rapido: 200 réplicas, sin modelo por plazo)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
import modelo_inundacion as mi  # noqa: E402

OBJ = mi.OBJETIVO
SALIDA = RAIZ / "data" / "modelo" / "resultados_realista"
JSON = RAIZ / "docs" / "evidencia" / "evaluacion_realista.json"
IMG = RAIZ / "docs" / "3_validacion" / "img"
ANIOS = list(range(2008, 2026))
RESERVADOS = {2017: "Niño Costero", 2019: "año normal", 2023: "ciclón Yaku"}
NINO = [2017, 2023]                      # Niño 1+2 medio febrero-abril >= +1 °C (base_distrito_semana.csv)
CLIMA = ["lluvia_7d", "lluvia_30d", "lluvia_90d", "lluvia_7d_prev", "lluvia_7d_anom", "lluvia_30d_anom",
         "lluvia_90d_anom", "caudal", "caudal_razon", "caudal_max_4s", "nino12", "nino34"]
MODELOS = ["lightgbm", "logistica_simple", "climatologia", "persistencia", "feb_abr"]
NOMBRE = {"lightgbm": "LightGBM", "logistica_simple": "Logística simple", "climatologia": "Climatología",
          "persistencia": "Persistencia (emergencia en t)", "feb_abr": "Siempre febrero-abril"}


# ------------------------------------------------------------------------------------------------ escenarios
def variables(b: pd.DataFrame, anio: int, escenario: str) -> pd.DataFrame:
    """Devuelve la base con las variables que se habrían conocido para predecir el año `anio`."""
    if escenario == "actual":
        return b
    d = b.copy()
    pasado = d[d["semana"] < pd.Timestamp(anio, 1, 1)]
    for c in ["lluvia_7d", "lluvia_30d", "lluvia_90d"]:
        normal = pasado.groupby(["clave", "semana_anio"])[c].mean().rename("_n")
        d[c + "_anom"] = d[c] - d.join(normal, on=["clave", "semana_anio"])["_n"]
    med = pasado.groupby(["rio", "semana_anio"])["caudal"].median().rename("_m")
    d["caudal_razon"] = d["caudal"] / d.join(med, on=["rio", "semana_anio"])["_m"].clip(lower=0.5)
    d = d.sort_values(["clave", "semana"])
    d[CLIMA] = d.groupby("clave")[CLIMA].shift(1)            # lo de la semana t aún no está completo el lunes
    d["log_lluvia_30d"] = np.log1p(d["lluvia_30d"].clip(lower=0))
    d["log_caudal_razon"] = np.log(d["caudal_razon"].clip(lower=0.05))
    return d


# --------------------------------------------------------------------------------------------- validación
def correr(b: pd.DataFrame, escenario: str) -> pd.DataFrame:
    filas = []
    b = b.sort_values(["clave", "semana"]).copy()
    g = b.groupby("clave")["emergencia_sem"]
    for h in range(1, 5):
        b[f"E_t{h}"] = g.shift(-h)                            # para medir cada plazo por separado
    for anio in ANIOS:
        d = variables(b, anio, escenario)
        et = d.dropna(subset=[OBJ])
        tr = et[et["semana"] < pd.Timestamp(anio, 1, 1) - pd.Timedelta(weeks=4)]
        te = et[et["anio"] == anio]
        tr, te = mi.con_historia(tr, te)
        sc, umbral = {}, {}
        p, m = mi.lightgbm(tr, te)
        sc["lightgbm"], umbral["lightgbm"] = p, np.quantile(m.predict_proba(tr[mi.VARIABLES])[:, 1], 0.95)
        p, m = mi.logistica(tr, te, mi.SIMPLE)
        ptr = m.predict_proba(tr[mi.SIMPLE].fillna(tr[mi.SIMPLE].median()))[:, 1]
        sc["logistica_simple"], umbral["logistica_simple"] = p, np.quantile(ptr, 0.95)
        sc["climatologia"], umbral["climatologia"] = mi.climatologia(tr, te), np.quantile(mi.climatologia(tr, tr), 0.95)
        tasa = tr.groupby(tr["emergencia_sem"].fillna(0))[OBJ].mean()
        sc["persistencia"], umbral["persistencia"] = te["emergencia_sem"].fillna(0).map(tasa).to_numpy(), tasa.max() - 1e-9
        sc["feb_abr"] = (te["semana"] + pd.Timedelta(days=14)).dt.month.between(2, 4).astype(float).to_numpy()
        umbral["feb_abr"] = 0.5
        f = te[["clave", "distrito", "departamento", "semana", "anio", "emergencia_sem", OBJ]
               + [f"E_t{h}" for h in range(1, 5)]].copy()
        for k in MODELOS:
            f[k] = sc[k]
            f["alerta_" + k] = sc[k] >= umbral[k]
        filas.append(f)
        print(f"  {escenario} {anio}: PR-AUC LightGBM {average_precision_score(te[OBJ], sc['lightgbm']):.3f} | "
              f"climatología {average_precision_score(te[OBJ], sc['climatologia']):.3f}", flush=True)
    return pd.concat(filas, ignore_index=True)


# ------------------------------------------------------------------------------------------------ métricas
def basicas(y, p, clim) -> dict:
    y, p = np.asarray(y), np.clip(np.asarray(p, float), 0, 1)
    bs, bc = brier_score_loss(y, p), brier_score_loss(y, np.asarray(clim))
    return {"pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p), "brier": bs,
            "bss_vs_climatologia": 1 - bs / bc, "positivos": int(y.sum()), "n": int(len(y)), "tasa": float(y.mean())}


def tabla(pr: pd.DataFrame, mask=None) -> dict:
    d = pr if mask is None else pr[mask]
    return {k: {c: round(float(v), 4) for c, v in basicas(d[OBJ], d[k], d["climatologia"]).items()} for k in MODELOS}


def por_anio(pr: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for a, d in pr.groupby("anio"):
        for k in MODELOS:
            filas.append({"anio": a, "modelo": k, **basicas(d[OBJ], d[k], d["climatologia"])})
    return pd.DataFrame(filas)


def calibracion(pr: pd.DataFrame, k: str, n=10) -> list[dict]:
    q = pd.qcut(pr[k].rank(method="first"), n, labels=False)
    t = pr.groupby(q).agg(prob_media=(k, "mean"), frecuencia_real=(OBJ, "mean"), n=(OBJ, "size"))
    return t.round(4).to_dict(orient="records")


def episodios(pr: pd.DataFrame) -> pd.DataFrame:
    """Un episodio empieza en la semana s con emergencia si no hubo emergencia en las 2 semanas anteriores."""
    out = []
    for c, d in pr.sort_values("semana").groupby("clave"):
        e = d.set_index("semana")["emergencia_sem"]
        for s in e[e == 1].index:
            antes = [s - pd.Timedelta(weeks=k) for k in (1, 2)]
            if all(e.get(a, 0) != 1 for a in antes):
                out.append({"clave": c, "inicio": s, "departamento": d["departamento"].iloc[0]})
    return pd.DataFrame(out)


def por_evento(pr: pd.DataFrame, ep: pd.DataFrame) -> dict:
    """¿Hubo alerta en alguna de las 4 semanas anteriores al inicio? Anticipación = semanas entre la primera alerta
    de esa ventana y el inicio. Falsas alarmas = semanas en alerta sin emergencia en las 4 semanas siguientes."""
    idx = pr.set_index(["clave", "semana"])
    res = {}
    temporadas = pr[pr["semana"].dt.month.between(1, 4)].groupby(["clave", "anio"]).ngroups
    for k in MODELOS:
        al = idx["alerta_" + k]
        avisados, antic = 0, []
        for r in ep.itertuples():
            v = [al.get((r.clave, r.inicio - pd.Timedelta(weeks=w)), False) for w in range(4, 0, -1)]
            if any(v):
                avisados += 1
                antic.append(4 - int(np.argmax(v)))
        a = pr["alerta_" + k]
        fa = int((a & (pr[OBJ] == 0)).sum())
        res[k] = {"episodios": len(ep), "avisados": avisados, "pct_avisados": round(100 * avisados / len(ep), 1),
                  "anticipacion_mediana_semanas": float(np.median(antic)) if antic else None,
                  "semanas_en_alerta": int(a.sum()), "pct_semanas_en_alerta": round(100 * a.mean(), 1),
                  "precision_alertas": round(float(pr.loc[a, OBJ].mean()), 3) if a.any() else None,
                  "falsas_alarmas": fa, "falsas_alarmas_por_distrito_temporada": round(fa / temporadas, 2)}
    return res


def por_plazo(pr: pd.DataFrame) -> dict:
    out = {}
    for h in range(1, 5):
        d = pr.dropna(subset=[f"E_t{h}"])
        out[f"h{h}"] = {k: round(float(average_precision_score(d[f"E_t{h}"], d[k])), 4) for k in MODELOS}
        out[f"h{h}"]["tasa"] = round(float(d[f"E_t{h}"].mean()), 4)
    return out


# ----------------------------------------------------------------------------------------------- bootstrap
def bootstrap(pr: pd.DataFrame, bloque: list[str], reps: int, semilla=2026) -> dict:
    rng = np.random.default_rng(semilla)
    grupos = [g.to_numpy() for _, g in pr.groupby(bloque).indices.items()] if False else \
        list(pr.groupby(bloque).indices.values())
    y = pr[OBJ].to_numpy()
    cols = {k: pr[k].to_numpy() for k in ["lightgbm", "logistica_simple", "climatologia"]}
    dif = {k: [] for k in ["lightgbm", "logistica_simple"]}
    bss = {k: [] for k in dif}
    for _ in range(reps):
        i = np.concatenate([grupos[j] for j in rng.integers(0, len(grupos), len(grupos))])
        yy = y[i]
        if yy.sum() == 0:
            continue
        base = average_precision_score(yy, cols["climatologia"][i])
        bc = np.mean((cols["climatologia"][i] - yy) ** 2)
        for k in dif:
            dif[k].append(average_precision_score(yy, cols[k][i]) - base)
            bss[k].append(1 - np.mean((cols[k][i] - yy) ** 2) / bc)
    r = {}
    for k in dif:
        a, s = np.array(dif[k]), np.array(bss[k])
        r[k] = {"dif_pr_auc_media": round(float(a.mean()), 4),
                "dif_pr_auc_ic95": [round(float(np.quantile(a, .025)), 4), round(float(np.quantile(a, .975)), 4)],
                "prob_supera_climatologia": round(float((a > 0).mean()), 3),
                "bss_ic95": [round(float(np.quantile(s, .025)), 4), round(float(np.quantile(s, .975)), 4)]}
    return {"bloques": " x ".join(bloque), "n_bloques": len(grupos), "replicas": reps, "modelos": r}


# ---------------------------------------------------------------------------------------- modelo por plazo
def plazo_sin_emergencia_t() -> dict:
    import prediccion_semanal as ps
    modelo = "logística con variables del aviso"
    out, original, panel = {}, list(ps.CON_AVISO), ps.panel
    sin = [c for c in original if c != "emergencia_t"]
    clima = ["log_lluvia_30d", "log_caudal_razon", "nino12", "lluvia_sobre_p95", "lluvia_sobre_p99", "log_lluvia_7d",
             "caudal_sobre_p95"]

    def desfasado():                                      # clima, río y mar de la semana t-1 (llegan con retraso)
        p = panel()
        p[clima] = p.groupby("clave")[clima].shift(1)
        return p

    for nombre, cols, fn in [("con_emergencia_t", original, panel), ("sin_emergencia_t", sin, panel),
                             ("sin_emergencia_t_y_clima_desfasado_1_semana", sin, desfasado)]:
        ps.CON_AVISO, ps.panel = cols, fn
        _, _, res = ps.correr()
        r = res.set_index(["h", "modelo"])
        out[nombre] = {f"h{h}": {"pr_auc": round(float(r.loc[(h, modelo), "pr_auc"]), 4),
                                 "pr_auc_climatologia": round(float(r.loc[(h, "climatología distrito y mes"), "pr_auc"]), 4),
                                 "bss": round(float(r.loc[(h, modelo), "bss"]), 4),
                                 "bss_climatologia": round(float(r.loc[(h, "climatología distrito y mes"), "bss"]), 4)}
                       for h in ps.PLAZOS}
    ps.CON_AVISO, ps.panel = original, panel
    return out


# --------------------------------------------------------------------------------------------------- gráficos
def graficos(pa: pd.DataFrame, pr: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    IMG.mkdir(parents=True, exist_ok=True)
    azul, naranja, gris, tinta = "#2a78d6", "#eb6834", "#8a8984", "#52514e"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#c9c8c2", "axes.labelcolor": tinta, "xtick.color": tinta, "ytick.color": tinta})
    # 1. PR-AUC por año
    t = pa.pivot(index="anio", columns="modelo", values="pr_auc")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t.index, t["lightgbm"], color=azul, lw=2, marker="o", ms=5, label="LightGBM (escenario realista)")
    ax.plot(t.index, t["climatologia"], color=naranja, lw=2, marker="o", ms=5, label="Climatología")
    for a in RESERVADOS:
        ax.axvline(a, color="#e4e3dd", lw=6, zorder=0)
        ax.text(a, ax.get_ylim()[1], f"{a}\n{RESERVADOS[a]}", ha="center", va="bottom", fontsize=8, color=tinta)
    ax.set_ylabel("PR-AUC del año"); ax.set_xticks(t.index); ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", color="#eeede8"); ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(IMG / "realidad_pr_auc_por_anio.png", dpi=150); plt.close(fig)
    # 2. calibración
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color=gris, lw=1, ls="--", label="Calibración perfecta")
    for k, c in [("lightgbm", azul), ("climatologia", naranja)]:
        cal = pd.DataFrame(calibracion(pr, k))
        ax.plot(cal["prob_media"], cal["frecuencia_real"], color=c, lw=2, marker="o", ms=6, label=NOMBRE[k])
    ax.set_xlabel("Probabilidad que dio el modelo"); ax.set_ylabel("Frecuencia real de Y = 1")
    m = max(pr["lightgbm"].quantile(.99), pr[OBJ].groupby(pd.qcut(pr["lightgbm"].rank(method="first"), 10), observed=True).mean().max())
    ax.set_xlim(0, min(1, m * 1.15)); ax.set_ylim(0, min(1, m * 1.15))
    ax.grid(color="#eeede8"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(IMG / "realidad_calibracion.png", dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------------------------------- main
def main(rapido: bool) -> dict:
    SALIDA.mkdir(parents=True, exist_ok=True)
    b = mi.cargar("sinpad")
    out = {"pregunta": "¿El modelo avisa mejor que la climatología con solo lo que se sabía cada lunes?"}
    res = {}
    for esc in ["actual", "realista"]:
        res[esc] = correr(b, esc)
        out[f"global_{esc}"] = tabla(res[esc])
    pr = res["realista"]
    pr.to_csv(SALIDA / "predicciones_realista.csv.gz", index=False, float_format="%.5f")
    pa = por_anio(pr)
    pa.to_csv(SALIDA / "metricas_por_anio_realista.csv", index=False, float_format="%.4f")
    t = pa.pivot(index="anio", columns="modelo", values="pr_auc")
    out["anios_que_gana_a_climatologia"] = {k: int((t[k] > t["climatologia"]).sum()) for k in ["lightgbm", "logistica_simple"]}
    out["anios_evaluados"] = len(t)
    out["pr_auc_medio_anual"] = {k: round(float(t[k].mean()), 4) for k in MODELOS}
    out["anios_reservados"] = {}
    ep = episodios(pr)
    for a, nombre in RESERVADOS.items():
        m = pr["anio"] == a
        out["anios_reservados"][str(a)] = {"nombre": nombre, "metricas": tabla(pr, m),
                                           "eventos": por_evento(pr[m], ep[ep["inicio"].dt.year == a])}
    out["calibracion"] = {k: calibracion(pr, k) for k in ["lightgbm", "logistica_simple", "climatologia"]}
    out["por_evento"] = por_evento(pr, ep)
    out["por_plazo"] = por_plazo(pr)
    out["por_departamento"] = {d: tabla(pr, pr["departamento"] == d) for d in ["Tumbes", "Piura"]}
    out["por_tipo_de_anio"] = {"nino_costero_2017_2023": tabla(pr, pr["anio"].isin(NINO)),
                               "otros_anios": tabla(pr, ~pr["anio"].isin(NINO))}
    reps = 200 if rapido else 1000
    out["bootstrap_distrito_anio"] = bootstrap(pr, ["clave", "anio"], reps)
    out["bootstrap_anio"] = bootstrap(pr, ["anio"], 2 * reps)
    if not rapido:
        print("Modelo por plazo con y sin emergencia_t (tarda unos minutos)...", flush=True)
        out["modelo_por_plazo"] = plazo_sin_emergencia_t()
    graficos(pa, pr)
    JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    g, r = out["global_actual"], out["global_realista"]
    print("\nPR-AUC global (actual -> realista):")
    for k in MODELOS:
        print(f"  {NOMBRE[k]:32s} {g[k]['pr_auc']:.3f} -> {r[k]['pr_auc']:.3f}   BSS {r[k]['bss_vs_climatologia']:+.3f}")
    for k, v in out["bootstrap_anio"]["modelos"].items():
        print(f"  {k}: diferencia con climatología {v['dif_pr_auc_media']:+.3f} IC95 {v['dif_pr_auc_ic95']}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    main(ap.parse_args().rapido)
