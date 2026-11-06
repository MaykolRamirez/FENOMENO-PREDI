"""Imágenes Sentinel-2 antes y ahora de las zonas donde El Niño golpea más, para el comparador del tablero.

Uso:
    python src/imagenes_satelite.py

Para cada zona busca en Microsoft Planetary Computer (STAC, sin clave) dos escenas Sentinel-2 L2A:
    ahora  la menos nubosa de los últimos 30 días
    antes  la menos nubosa de la misma época un año antes (referencia del terreno sin las lluvias de este año)
Solo se aceptan escenas cuyo contorno cubre el centro y al menos 3 de las 4 esquinas de la zona (evita franjas negras grandes).
De cada escena guarda tres vistas: color real de 1.600 x 1.600 px y vegetación y agua de 800 x 800 px: color real, vegetación (NDVI) y agua (NDWI).

Salidas en dashboard/data/sat/:  {zona}_{antes|ahora}_{rgb|ndvi|agua}.jpg  y  satelite.json
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone

import requests

from config import RAIZ

SAT = RAIZ / "dashboard" / "data" / "sat"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
DATA = "https://planetarycomputer.microsoft.com/api/data/v1/item/bbox/{x0},{y0},{x1},{y1}/{px}x{px}.{fmt}"
VENTANA_AHORA = 30     # días hacia atrás
VENTANA_ANTES = 20     # ± días alrededor de la misma fecha un año antes
MAX_NUBES = 85         # % de nubes de la escena (más que esto no sirve para comparar)

# (id, nombre, oeste, sur, este, norte, por qué se vigila). Cajas de 0,2° (~22 km) que caben en una sola escena Sentinel-2.
ZONAS = [
    ("tumbes", "Tumbes", -80.55, -3.90, -80.35, -3.70, "Llanura de Zarumilla y bajo Tumbes"),
    ("piura", "Piura", -80.80, -5.30, -80.60, -5.10, "Valle del Piura"),
    ("lambayeque", "Lambayeque", -79.90, -6.80, -79.70, -6.60, "Valle Chancay-Lambayeque"),
    ("trujillo", "Trujillo", -79.10, -8.20, -78.90, -8.00, "Valles de Moche y Chicama"),
    ("lima", "Lima", -77.10, -12.10, -76.90, -11.90, "Ríos Rímac y Chillón"),
    ("huaraz", "Huaraz", -77.60, -9.60, -77.40, -9.40, "Callejón de Huaylas (quebradas)"),
]
VISTAS = {
    "rgb": ("jpg", {"assets": ["B04", "B03", "B02"], "asset_bidx": ["B04|1", "B03|1", "B02|1"],
                    "color_formula": "gamma RGB 1.9, saturation 1.7, sigmoidal RGB 11 0.18"}),
    "ndvi": ("jpg", {"expression": "(B08-B04)/(B08+B04)", "asset_as_band": "true",
                     "rescale": "-0.1,0.8", "colormap_name": "rdylgn"}),
    "agua": ("jpg", {"expression": "(B03-B08)/(B03+B08)", "asset_as_band": "true",
                     "rescale": "-0.6,0.4", "colormap_name": "ylgnbu"}),
}


def dentro(pt, poligono) -> bool:
    """Punto dentro de un anillo (trazado de rayos)."""
    x, y = pt
    ok = False
    for (x1, y1), (x2, y2) in zip(poligono, poligono[1:] + poligono[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            ok = not ok
    return ok


def cubre(item: dict, bbox: list[float], minimo: int = 3) -> bool:
    geom = item["geometry"]
    anillos = [geom["coordinates"][0]] if geom["type"] == "Polygon" else [p[0] for p in geom["coordinates"]]
    x0, y0, x1, y1 = bbox
    esquinas = [(x0, y0), (x0, y1), (x1, y0), (x1, y1), ((x0 + x1) / 2, (y0 + y1) / 2)]
    return any(dentro(esquinas[4], [tuple(p) for p in a]) and sum(dentro(c, [tuple(p) for p in a]) for c in esquinas[:4]) >= minimo
               for a in anillos)


def mejor_escena(bbox: list[float], desde: datetime, hasta: datetime) -> dict | None:
    cuerpo = {"collections": ["sentinel-2-l2a"], "bbox": bbox, "limit": 40,
              "datetime": f"{desde:%Y-%m-%d}/{hasta:%Y-%m-%d}",
              "query": {"eo:cloud_cover": {"lt": MAX_NUBES}},
              "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}]}
    for i in range(3):
        r = requests.post(STAC, json=cuerpo, timeout=90)
        if r.status_code == 200:
            break
        time.sleep(4 * (i + 1))
    else:
        r.raise_for_status()
    for minimo in (4, 3):            # primero escenas sin bordes negros; si no hay, se acepta una esquina vacía
        validas = [f for f in r.json()["features"] if cubre(f, bbox, minimo)]
        if validas:
            return validas[0]
    return None


def descargar(item_id: str, bbox: list[float], vista: str, destino) -> bool:
    fmt, extra = VISTAS[vista]
    url = DATA.format(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3], px=1600 if vista == "rgb" else 800, fmt=fmt)
    params = {"collection": "sentinel-2-l2a", "item": item_id, **extra}
    for i in range(3):
        r = requests.get(url, params=params, timeout=180)
        if r.status_code == 200:
            destino.write_bytes(r.content)
            return True
        time.sleep(4 * (i + 1))
    print(f"    aviso: {vista} de {item_id} falló ({r.status_code})")
    return False


if __name__ == "__main__":
    SAT.mkdir(parents=True, exist_ok=True)
    hoy = datetime.now(timezone.utc)
    ref = hoy - timedelta(days=365)
    meta = {"generado": hoy.strftime("%Y-%m-%d %H:%M UTC"), "zonas": []}
    for zid, nombre, o, s, e, n, nota in ZONAS:
        bbox = [o, s, e, n]
        print(f"{nombre}…")
        z = {"id": zid, "nombre": nombre, "nota": nota, "bbox": bbox, "escenas": {}}
        for etapa, (d, h) in {"ahora": (hoy - timedelta(days=VENTANA_AHORA), hoy),
                              "antes": (ref - timedelta(days=VENTANA_ANTES), ref + timedelta(days=VENTANA_ANTES))}.items():
            item = mejor_escena(bbox, d, h)
            if not item:
                print(f"  sin escena {etapa}")
                continue
            ok = {v: descargar(item["id"], bbox, v, SAT / f"{zid}_{etapa}_{v}.{VISTAS[v][0]}") for v in VISTAS}
            if all(ok.values()):
                z["escenas"][etapa] = {"fecha": item["properties"]["datetime"][:10], "nubes": round(item["properties"]["eo:cloud_cover"]),
                                       "id": item["id"]}
                print(f"  {etapa}: {z['escenas'][etapa]['fecha']} ({z['escenas'][etapa]['nubes']} % nubes)")
            time.sleep(1)
        if len(z["escenas"]) == 2:
            meta["zonas"].append(z)
    (SAT / "satelite.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(meta['zonas'])} zonas listas")
