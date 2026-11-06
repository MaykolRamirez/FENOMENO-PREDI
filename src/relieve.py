"""Relieve (elevación) de las zonas satelitales para la vista 3D del tablero.

Uso:
    python src/relieve.py

Descarga las teselas AWS Terrain Tiles (formato Terrarium, sin clave), las une, las remuestrea a una
malla regular en longitud y latitud (la misma proyección de las imágenes Sentinel-2 del tablero) y
guarda una malla de elevaciones en metros por zona.

El relieve no cambia de un día para otro: se genera una vez y se sube al repositorio.

Salidas en dashboard/data/dem/:  {zona}.json  con  {"n", "bbox", "min", "max", "h"}  (h = n x n enteros, fila 0 = norte)
"""
from __future__ import annotations

import io
import json
import math

import numpy as np
import requests
from PIL import Image

from config import RAIZ
from imagenes_satelite import ZONAS

DEM = RAIZ / "dashboard" / "data" / "dem"
URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
Z = 12          # ~38 m por píxel en el ecuador
N = 201         # puntos por lado de la malla (~110 m de separación en cajas de 22 km)


def tesela_xy(lon: float, lat: float, z: int) -> tuple[float, float]:
    """Coordenadas fraccionarias de tesela (Web Mercator) para una longitud y latitud."""
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def descargar(z: int, x: int, y: int) -> np.ndarray:
    r = requests.get(URL.format(z=z, x=x, y=y), timeout=60)
    r.raise_for_status()
    a = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"), dtype=np.float64)
    return a[..., 0] * 256.0 + a[..., 1] + a[..., 2] / 256.0 - 32768.0      # metros


def malla(bbox: list[float]) -> np.ndarray:
    o, s, e, n = bbox
    x0, y0 = tesela_xy(o, n, Z)           # esquina noroeste
    x1, y1 = tesela_xy(e, s, Z)           # esquina sureste
    tx0, tx1, ty0, ty1 = int(math.floor(x0)), int(math.floor(x1)), int(math.floor(y0)), int(math.floor(y1))
    mosaico = np.vstack([np.hstack([descargar(Z, tx, ty) for tx in range(tx0, tx1 + 1)]) for ty in range(ty0, ty1 + 1)])
    lons = np.linspace(o, e, N)
    lats = np.linspace(n, s, N)                                   # fila 0 = norte
    px = np.array([(tesela_xy(lon, n, Z)[0] - tx0) * 256 for lon in lons])
    py = np.array([(tesela_xy(o, lat, Z)[1] - ty0) * 256 for lat in lats])
    ix, iy = np.floor(px).astype(int), np.floor(py).astype(int)
    fx, fy = (px - ix)[None, :], (py - iy)[:, None]
    ix1, iy1 = np.minimum(ix + 1, mosaico.shape[1] - 1), np.minimum(iy + 1, mosaico.shape[0] - 1)
    h = (mosaico[np.ix_(iy, ix)] * (1 - fx) * (1 - fy) + mosaico[np.ix_(iy, ix1)] * fx * (1 - fy)
         + mosaico[np.ix_(iy1, ix)] * (1 - fx) * fy + mosaico[np.ix_(iy1, ix1)] * fx * fy)
    return h


if __name__ == "__main__":
    DEM.mkdir(parents=True, exist_ok=True)
    for zid, nombre, o, s, e, n, _ in ZONAS:
        h = malla([o, s, e, n])
        # El mar aparece como elevación 0 o ligeramente negativa en estas teselas: se recorta a 0.
        h = np.maximum(h, 0.0)
        h = np.rint(h).astype(int)
        datos = {"n": N, "bbox": [o, s, e, n], "min": int(h.min()), "max": int(h.max()), "h": h.ravel().tolist()}
        (DEM / f"{zid}.json").write_text(json.dumps(datos, separators=(",", ":")), encoding="utf-8")
        print(f"{nombre}: {h.min():.0f} a {h.max():.0f} m")
