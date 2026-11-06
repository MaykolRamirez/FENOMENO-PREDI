"""Pruebas rápidas: python tests/test_basico.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from calendario import fecha_publicacion, ultimo_mes_publicado, rezago_minimo
from descarga_bcrp import a_periodo, parsear_respuesta
from temperatura_mar import parsear_noaa_semanal, parsear_icen, a_mensual

# Calendario (julio 2026 se publicó el 15/09/2026)
assert fecha_publicacion("2026-07") == pd.Timestamp("2026-09-15")
assert str(ultimo_mes_publicado("2026-09-14")) == "2026-06"
assert str(ultimo_mes_publicado("2026-09-15")) == "2026-07"
assert rezago_minimo("2026-09", "2026-10-12") == 2   # retroproyección de setiembre
assert rezago_minimo("2026-11", "2026-11-03") == 3   # mes en curso, antes del 15

# Fechas del BCRP
assert str(a_periodo("Ene.1995")) == "1995-01"
assert str(a_periodo("Jul26")) == "2026-07"
assert str(a_periodo("Set.2026")) == "2026-09"

# Respuesta JSON de la API (estructura de BCRPData)
ejemplo = {"periods": [{"name": "Jul.2026", "values": ["3.5592604174", "n.d."]}]}
t = parsear_respuesta(ejemplo, ["PN01728AM", "PN01716AM"])
assert round(t.loc[0, "valor"], 2) == 3.56 and pd.isna(t.loc[1, "valor"])

# Archivo semanal de la NOAA (anomalía pegada cuando es negativa)
noaa = """ Weekly SST data starts week centered on 3Jan1990
                Nino1+2      Nino3        Nino34        Nino4
 Week          SST SSTA     SST SSTA     SST SSTA     SST SSTA
 03JAN1990     23.4-0.4     25.1-0.3     26.6 0.0     28.6 0.3
 10JAN1990     23.4-0.8     25.2-0.3     26.6 0.1     28.6 0.3
 07FEB1990     25.1 1.2     26.0 0.1     26.7 0.1     28.4 0.2
 14FEB1990     25.5 1.0     26.3 0.2     26.8 0.2     28.3 0.1
 07MAR1990     25.9 0.5     26.9 0.3     27.1 0.3     28.5 0.2
 07APR1990     25.0 0.9     27.5 0.4     27.6 0.4     28.6 0.2"""
s = parsear_noaa_semanal(noaa)
assert len(s) == 6 and s.loc[0, "anomalia_nino12"] == -0.4 and s.loc[2, "anomalia_nino12"] == 1.2
m = a_mensual(s)
assert round(m.loc[m.fecha == pd.Period("1990-01", "M"), "anomalia_nino12"].item(), 2) == -0.6
assert round(m.loc[3, "cambio_3m"], 2) == 1.5

# ICEN (año, mes, valor)
icen = parsear_icen("ICEN mensual\n1950 1 -1.24\n1950 2 -0.95\n")
assert len(icen) == 2 and icen.loc[0, "icen"] == -1.24

# Registro: guarda, no sobrescribe y rechaza meses ya publicados
import tempfile, config, registro
registro.REGISTROS = Path(tempfile.mkdtemp())
tabla = pd.DataFrame({"codigo": ["PN01728AM"], "p10": [1.0], "p50": [2.5], "p90": [4.0]})
mes_futuro = str(pd.Period(pd.Timestamp.today(), freq="M") + 1)
registro.registrar(mes_futuro, "prueba", tabla)
for kw in ({"mes": mes_futuro}, {"mes": "2026-07"}):
    try:
        registro.registrar(modelo="prueba", estimaciones=tabla, **kw)
        raise AssertionError("El registro debió bloquearse")
    except (FileExistsError, ValueError):
        pass

print("Todas las pruebas pasaron.")
