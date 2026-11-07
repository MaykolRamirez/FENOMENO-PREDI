"""Prueba prospectiva: guarda cada semana las predicciones ANTES de que ocurra lo que predicen.

Es la única prueba sin ninguna fuga de información posible (docs/3_validacion/24_MEJORAS_TRAS_EL_TEST.md): las
predicciones quedan en git con su fecha y la huella (SHA-256) del código del modelo, así que nadie puede cambiarlas
después. Cuando el SINPAD publique las emergencias de la temporada (enero-abril 2027), se comparan con lo que pasó.

    registrar   lee dashboard/data/prediccion_semanal.json y dashboard/data/riesgo.json y agrega una fila por distrito
                y plazo a data/prospectivo/predicciones.csv (si esa semana base ya está registrada, no hace nada)
    --evaluar   compara las predicciones registradas con el SINPAD (data/modelo/sinpad/sinpad_semanal.csv) en las
                semanas que ya tienen dato, frente a la climatología registrada en el mismo momento

Regla para que la prueba valga: no cambiar src/prediccion_semanal.py ni src/modelo_inundacion.py entre enero y abril de
2027. Si cambian, la huella cambia y la evaluación lo separa.
Uso: python src/registro_prospectivo.py [--evaluar]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import DASH_DATA, DATA  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
CARPETA = DATA / "prospectivo"
REGISTRO = CARPETA / "predicciones.csv"
CODIGO = [RAIZ / "src" / "prediccion_semanal.py", RAIZ / "src" / "modelo_inundacion.py"]


def huella() -> str:
    h = hashlib.sha256()
    for f in CODIGO:
        h.update(f.read_bytes().replace(b"\r\n", b"\n"))      # igual en Windows y en GitHub Actions
    return h.hexdigest()[:16]


def plano(x: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode().upper() if c.isalnum())


def filas() -> pd.DataFrame:
    p = json.loads((DASH_DATA / "prediccion_semanal.json").read_text(encoding="utf-8"))
    r = json.loads((DASH_DATA / "riesgo.json").read_text(encoding="utf-8"))
    riesgo = {plano(d["departamento"]) + plano(d["provincia"]) + plano(d["distrito"]): d for d in r["distritos_hoy"]}
    ahora = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    out = []
    for d in p["distritos"]:
        rr = riesgo.get(plano(d["departamento"]) + plano(d["provincia"]) + plano(d["distrito"]), {})
        for h in (1, 2, 3, 4):
            out.append({"registrado_utc": ahora, "huella_modelo": huella(), "semana_base": p["semana_base"],
                        "semana_datos": p.get("semana_datos"), "h": h, "semana_objetivo": p["semana_objetivo"][f"h{h}"],
                        "ubigeo": d["ubigeo"], "departamento": d["departamento"], "provincia": d["provincia"],
                        "distrito": d["distrito"], "prob": d[f"prob_h{h}"], "climatologia": d.get(f"clim_h{h}"),
                        "riesgo_4_semanas": rr.get("riesgo"), "alerta_4_semanas": rr.get("alerta"),
                        "semana_riesgo_4_semanas": r.get("semana_actual"),
                        "fuera_de_rango": "; ".join(v["variable"] for v in p.get("fuera_de_rango", []))})
    return pd.DataFrame(out)


def registrar() -> None:
    CARPETA.mkdir(parents=True, exist_ok=True)
    nuevo = filas()
    if REGISTRO.exists():
        viejo = pd.read_csv(REGISTRO, dtype={"ubigeo": str})
        ya = set(zip(viejo.semana_base, viejo.huella_modelo))
        if (nuevo.semana_base.iloc[0], nuevo.huella_modelo.iloc[0]) in ya:
            print(f"Semana base {nuevo.semana_base.iloc[0]} ya registrada con este modelo: sin cambios")
            return
        nuevo = pd.concat([viejo, nuevo], ignore_index=True)
    nuevo.to_csv(REGISTRO, index=False)
    print(f"Registradas {len(nuevo):,} predicciones (semana base {nuevo.semana_base.iloc[-1]}, modelo {huella()})")


def evaluar() -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss
    from etiqueta_sinpad import normal
    reg = pd.read_csv(REGISTRO, dtype={"ubigeo": str}, parse_dates=["semana_objetivo"])
    s = pd.read_csv(DATA / "modelo" / "sinpad" / "sinpad_semanal.csv", parse_dates=["semana"])
    reg["clave"] = (reg.departamento.map(plano) + reg.distrito.map(plano))
    reg["prov_n"] = reg.provincia.map(normal)
    reg = reg.merge(s[["clave", "prov_n", "semana", "eventos"]].rename(columns={"semana": "semana_objetivo"}),
                    on=["clave", "prov_n", "semana_objetivo"], how="left")
    hay = reg.semana_objetivo <= s.semana.max()
    reg = reg[hay].assign(y=lambda d: (d.eventos.fillna(0) > 0).astype(int))
    out = {"semanas_con_dato_hasta": str(s.semana.max().date()), "predicciones_evaluables": len(reg)}
    for (h, hu), d in reg.groupby(["h", "huella_modelo"]):
        if d.y.sum() == 0:
            out[f"h{h}_{hu}"] = {"n": len(d), "positivos": 0}
            continue
        out[f"h{h}_{hu}"] = {"n": len(d), "positivos": int(d.y.sum()),
                             "pr_auc_modelo": round(average_precision_score(d.y, d.prob), 4),
                             "pr_auc_climatologia": round(average_precision_score(d.y, d.climatologia), 4),
                             "brier_modelo": round(brier_score_loss(d.y, d.prob), 4),
                             "brier_climatologia": round(brier_score_loss(d.y, d.climatologia), 4)}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluar", action="store_true")
    evaluar() if ap.parse_args().evaluar else registrar()
