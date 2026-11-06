"""Registro de estimaciones en tiempo real (antes de la cifra oficial del INEI).

Cada registro es un CSV nuevo en registros/nowcasts/. Nunca se sobrescribe:
si el archivo existe, el script se detiene. Después de registrar, se hace
commit y push para que GitHub deje constancia de la fecha.

Uso desde Python:
    from registro import registrar
    registrar(mes="2026-09", modelo="AR + electricidad", estimaciones=tabla)

donde `tabla` tiene columnas: codigo, p10, p50, p90.
"""
from __future__ import annotations

from datetime import datetime
import warnings

import pandas as pd

from calendario import fecha_publicacion, rezago_minimo
from config import REGISTROS, SERIES_OBJETIVO


def registrar(mes: str, modelo: str, estimaciones: pd.DataFrame, version: int = 1,
              nota: str = "") -> str:
    ahora = datetime.now()
    publicacion = fecha_publicacion(mes)
    if ahora >= publicacion:
        raise ValueError(
            f"La cifra de {mes} ya se publicó ({publicacion:%d/%m/%Y}). "
            "Un registro posterior no sirve como prueba en tiempo real."
        )
    if (publicacion - ahora).days < 3:
        warnings.warn(
            f"Faltan menos de 3 días para la publicación prevista ({publicacion:%d/%m/%Y}). "
            "Si el día 15 cae en fin de semana, el INEI puede publicar antes: "
            "verifique que la cifra aún no exista."
        )
    faltan = {"codigo", "p10", "p50", "p90"} - set(estimaciones.columns)
    if faltan:
        raise ValueError(f"Faltan columnas: {faltan}")

    REGISTROS.mkdir(parents=True, exist_ok=True)
    archivo = REGISTROS / f"nowcast_{mes}_v{version}.csv"
    if archivo.exists():
        raise FileExistsError(f"{archivo.name} ya existe. Use version={version + 1}; no se sobrescribe.")

    salida = estimaciones.copy()
    salida["serie"] = salida["codigo"].map(SERIES_OBJETIVO)
    salida["mes_estimado"] = mes
    salida["fecha_registro"] = ahora.strftime("%Y-%m-%d %H:%M")
    salida["fecha_publicacion_inei"] = publicacion.strftime("%Y-%m-%d")
    salida["rezago_k"] = rezago_minimo(mes, ahora)
    salida["modelo"] = modelo
    salida["version"] = version
    salida["nota"] = nota
    salida.to_csv(archivo, index=False)
    print(f"Registrado {archivo.name}. Ahora: git add, git commit y git push.")
    return str(archivo)
