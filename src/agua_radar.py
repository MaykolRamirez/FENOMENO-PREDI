"""Agua detectada con radar Sentinel-1 (atraviesa las nubes) para las zonas del tablero.

Uso:
    python src/agua_radar.py

Para cada zona busca las pasadas Sentinel-1 RTC de los últimos 30 días (Microsoft Planetary Computer, sin clave)
y las compara con una referencia de la misma época un año antes y de la MISMA órbita (misma geometría de
observación, para que la diferencia no sea un efecto del ángulo del satélite).

    agua nueva         píxel que ahora es agua y en la referencia era tierra
    agua permanente    píxel que es agua ahora y en la referencia (mar, ríos, lagunas, embalses)

Cómo se decide que un píxel es agua:
    - retrodispersión VV suavizada (promedio 3 x 3) menor que -18 dB (el agua en calma refleja casi nada hacia el radar)
    - tierra en la referencia: mayor que -15 dB
    - el agua nueva se limpia con una apertura 3 x 3 para quitar píxeles aislados (ruido del radar)
    - se descartan los píxeles sin datos (borde de la escena) y las pendientes de más de 10 grados, donde la
      sombra del relieve también se ve oscura y se confundiría con agua

Salidas en dashboard/data/agua/:  {zona}_{fecha}_{orbita}.png (capa transparente)  y  agua.json
"""
from __future__ import annotations

import io
import json
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import requests
from PIL import Image

from config import RAIZ
from imagenes_satelite import STAC, ZONAS

SALIDA = RAIZ / "dashboard" / "data" / "agua"
DEM = RAIZ / "dashboard" / "data" / "dem"
DATA = "https://planetarycomputer.microsoft.com/api/data/v1/item/bbox/{x0},{y0},{x1},{y1}/{px}x{px}.png"
PX = 800
DIAS = 30
UMBRAL_AGUA = -18.0       # dB
UMBRAL_TIERRA = -15.0     # dB
PENDIENTE_MAX = 10.0      # grados
SIN_DATO = -29.0          # dB: por debajo es relleno de la escena, no una medición


def suavizar(a: np.ndarray) -> np.ndarray:
    """Promedio 3 x 3: reduce el moteado (speckle) propio del radar."""
    p = np.pad(a, 1, mode="edge")
    return sum(p[i:i + a.shape[0], j:j + a.shape[1]] for i in range(3) for j in range(3)) / 9.0


def remuestrear(h: np.ndarray, px: int) -> np.ndarray:
    """Pasa una malla n x n a px x px con interpolación lineal."""
    n = h.shape[0]
    x = np.linspace(0, n - 1, px)
    f = np.arange(n)
    por_filas = np.array([np.interp(x, f, fila) for fila in h])
    return np.array([np.interp(x, f, col) for col in por_filas.T]).T


def pendiente_grados(h: np.ndarray, ancho_m: float, alto_m: float) -> np.ndarray:
    """Pendiente en grados de una malla de elevaciones en metros (fila 0 = norte)."""
    dy, dx = np.gradient(h, alto_m / h.shape[0], ancho_m / h.shape[1])
    return np.degrees(np.arctan(np.hypot(dx, dy)))


def abrir(m: np.ndarray) -> np.ndarray:
    """Apertura 3 x 3 (erosión y luego dilatación): borra píxeles aislados, que son ruido del radar y no inundaciones."""
    def vecindad(a, f):
        p = np.pad(a, 1, mode="constant", constant_values=False)
        return f([p[i:i + a.shape[0], j:j + a.shape[1]] for i in range(3) for j in range(3)], axis=0)
    return vecindad(vecindad(m, np.all), np.any)


def mascaras(ahora_db: np.ndarray, ref_db: np.ndarray, pendiente: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (agua nueva, agua permanente) como máscaras booleanas."""
    valido = (ahora_db > SIN_DATO) & (ref_db > SIN_DATO) & (pendiente <= PENDIENTE_MAX)
    agua = (ahora_db < UMBRAL_AGUA) & valido
    permanente = agua & (ref_db < UMBRAL_AGUA)
    nueva = abrir(agua & (ref_db >= UMBRAL_TIERRA))
    return nueva, permanente


def buscar(bbox: list[float], desde: datetime, hasta: datetime) -> list[dict]:
    cuerpo = {"collections": ["sentinel-1-rtc"], "bbox": bbox, "limit": 100,
              "datetime": f"{desde:%Y-%m-%d}/{hasta:%Y-%m-%d}"}
    for i in range(3):
        r = requests.post(STAC, json=cuerpo, timeout=90)
        if r.status_code == 200:
            return sorted(r.json()["features"], key=lambda f: f["properties"]["datetime"])
        time.sleep(4 * (i + 1))
    r.raise_for_status()


def descargar_db(item_id: str, bbox: list[float]) -> np.ndarray | None:
    """Retrodispersión VV en dB (escala 8 bits entre -30 y +5 dB, convertida de vuelta)."""
    url = DATA.format(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3], px=PX)
    params = {"collection": "sentinel-1-rtc", "item": item_id, "assets": "vv", "expression": "10*log10(vv)",
              "asset_as_band": "true", "rescale": "-30,5", "format": "png"}
    for i in range(3):
        r = requests.get(url, params=params, timeout=180)
        if r.status_code == 200:
            a = np.asarray(Image.open(io.BytesIO(r.content)))
            a = a[..., 0] if a.ndim == 3 else a
            return -30.0 + a.astype(float) / 255.0 * 35.0
        time.sleep(4 * (i + 1))
    print(f"    aviso: no se pudo descargar {item_id} ({r.status_code})")
    return None


def una_por_pasada(items: list[dict]) -> dict:
    """Agrupa por (fecha, órbita): una misma pasada puede venir partida en varios cuadros."""
    grupos: dict = {}
    for f in items:
        clave = (f["properties"]["datetime"][:10], f["properties"].get("sat:relative_orbit"))
        grupos.setdefault(clave, f)
    return grupos


def capa(nueva: np.ndarray, permanente: np.ndarray) -> Image.Image:
    rgba = np.zeros(nueva.shape + (4,), dtype=np.uint8)
    rgba[permanente] = (45, 115, 230, 150)
    rgba[nueva] = (0, 225, 255, 235)
    return Image.fromarray(rgba, "RGBA")


if __name__ == "__main__":
    SALIDA.mkdir(parents=True, exist_ok=True)
    hoy = datetime.now(timezone.utc)
    ref_ini, ref_fin = hoy - timedelta(days=365 + 25), hoy - timedelta(days=365 - 25)
    meta = {"generado": hoy.strftime("%Y-%m-%d %H:%M UTC"), "umbral_agua_db": UMBRAL_AGUA, "umbral_tierra_db": UMBRAL_TIERRA,
            "pendiente_max": PENDIENTE_MAX, "dias": DIAS, "zonas": {}}
    for zid, nombre, o, s, e, n, _ in ZONAS:
        bbox = [o, s, e, n]
        print(f"{nombre}…")
        dem_ruta = DEM / f"{zid}.json"
        if not dem_ruta.exists():
            print("  sin relieve; se omite (ejecuta src/relieve.py)")
            continue
        d = json.loads(dem_ruta.read_text(encoding="utf-8"))
        h = remuestrear(np.array(d["h"], dtype=float).reshape(d["n"], d["n"]), PX)
        ancho = (e - o) * 111320.0 * np.cos(np.radians((s + n) / 2))
        alto = (n - s) * 110574.0
        pend = pendiente_grados(h, ancho, alto)
        ha_px = ancho / PX * alto / PX / 10000.0

        ahora = una_por_pasada(buscar(bbox, hoy - timedelta(days=DIAS), hoy))
        refs_por_orbita: dict = {}
        for (fecha, orb), f in una_por_pasada(buscar(bbox, ref_ini, ref_fin)).items():
            refs_por_orbita.setdefault(orb, []).append(f)
        referencia: dict = {}
        escenas = []
        for (fecha, orb), f in sorted(ahora.items()):
            if orb not in referencia:
                lista = refs_por_orbita.get(orb, [])
                muestras = [m for m in (descargar_db(x["id"], bbox) for x in lista[::max(1, len(lista) // 3)][:3]) if m is not None]
                referencia[orb] = np.median(np.stack([suavizar(m) for m in muestras]), axis=0) if muestras else None
            if referencia[orb] is None:
                print(f"  {fecha}: sin referencia para la órbita {orb}")
                continue
            a = descargar_db(f["id"], bbox)
            if a is None:
                continue
            nueva, perm = mascaras(suavizar(a), referencia[orb], pend)
            archivo = f"{zid}_{fecha}_{orb}.png"
            capa(nueva, perm).save(SALIDA / archivo, optimize=True)
            escenas.append({"fecha": fecha, "orbita": orb, "nueva_ha": round(float(nueva.sum() * ha_px), 1),
                            "permanente_ha": round(float(perm.sum() * ha_px), 1), "archivo": archivo})
            print(f"  {fecha} órbita {orb}: nueva {escenas[-1]['nueva_ha']} ha, permanente {escenas[-1]['permanente_ha']} ha")
            time.sleep(0.5)
        meta["zonas"][zid] = {"nombre": nombre, "escenas": escenas}
    (SALIDA / "agua.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("listo")
