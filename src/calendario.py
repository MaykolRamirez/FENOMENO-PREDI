"""Calendario de publicación: qué cifra oficial existe en cada fecha.

Es la pieza que evita la fuga de información. Antes de usar el valor de un
mes como predictora, el modelo pregunta si ese mes ya estaba publicado en la
fecha de estimación.
"""
from __future__ import annotations

import pandas as pd

from config import DIA_PUBLICACION_INEI, MESES_REZAGO_PUBLICACION


def fecha_publicacion(mes: str | pd.Period) -> pd.Timestamp:
    """Fecha en que el INEI publica la cifra del mes indicado.

    >>> fecha_publicacion("2026-07")
    Timestamp('2026-09-15 00:00:00')
    """
    periodo = pd.Period(mes, freq="M")
    mes_pub = periodo + MESES_REZAGO_PUBLICACION
    return mes_pub.to_timestamp() + pd.Timedelta(days=DIA_PUBLICACION_INEI - 1)


def ultimo_mes_publicado(fecha: str | pd.Timestamp) -> pd.Period:
    """Último mes con cifra oficial publicada en la fecha dada.

    Antes del día 15 la última cifra es la de t-3; desde el día 15, la de t-2.

    >>> str(ultimo_mes_publicado("2026-09-10"))
    '2026-06'
    >>> str(ultimo_mes_publicado("2026-09-15"))
    '2026-07'
    """
    fecha = pd.Timestamp(fecha)
    mes_actual = pd.Period(fecha, freq="M")
    if fecha.day >= DIA_PUBLICACION_INEI:
        return mes_actual - MESES_REZAGO_PUBLICACION
    return mes_actual - MESES_REZAGO_PUBLICACION - 1


def rezago_minimo(mes_objetivo: str | pd.Period, fecha: str | pd.Timestamp) -> int:
    """Rezago k_d: cuántos meses separan el mes estimado de la última cifra publicada.

    >>> rezago_minimo("2026-08", "2026-09-20")   # agosto, estimado el 20 de setiembre
    1
    >>> rezago_minimo("2026-09", "2026-10-12")   # setiembre, estimado el 12 de octubre
    2
    """
    objetivo = pd.Period(mes_objetivo, freq="M")
    return (objetivo - ultimo_mes_publicado(fecha)).n


if __name__ == "__main__":
    import doctest

    resultado = doctest.testmod(verbose=False)
    print(f"Pruebas del calendario: {resultado.attempted - resultado.failed}/{resultado.attempted} correctas")
