"""Series de tiempo sintéticas para probar modelos de pronóstico.

Viene del cuaderno «Series temporales sintéticas» (08_optimizacion_y_series).
Cambios: solo NumPy (sin pandas ni gráficos) y semilla opcional para que los
resultados se puedan repetir.
"""

from __future__ import annotations

import numpy as np


def _rng(semilla: int | np.random.Generator | None) -> np.random.Generator:
    return semilla if isinstance(semilla, np.random.Generator) else np.random.default_rng(semilla)


def fechas(n: int, inicio: str = "2023-01-01") -> np.ndarray:
    """Fechas diarias consecutivas (``datetime64[D]``)."""
    return np.datetime64(inicio, "D") + np.arange(n)


def serie_tendencia(n: int = 100, pendiente: float = 0.5, ruido: float = 5.0, semilla=None) -> np.ndarray:
    """Línea recta con ruido: ``pendiente * t + ruido``."""
    return np.arange(n) * pendiente + _rng(semilla).standard_normal(n) * ruido


def serie_estacional(n: int = 365, periodo: float = 30, amplitud: float = 10.0,
                     tendencia_final: float = 5.0, ruido: float = 0.5, semilla=None) -> np.ndarray:
    """Onda seno (ciclo de ``periodo`` puntos) + tendencia lineal + ruido."""
    t = np.arange(n)
    return (np.sin(2 * np.pi * t / periodo) * amplitud
            + np.linspace(0, tendencia_final, n)
            + _rng(semilla).standard_normal(n) * ruido)


def serie_ar(n: int, coeficientes: list[float], intercepto: float = 0.0, ruido: float = 1.0, semilla=None) -> np.ndarray:
    """Proceso autorregresivo AR(p): cada valor depende de los ``p`` anteriores.

    ``x[t] = intercepto + c1*x[t-1] + ... + cp*x[t-p] + ruido``
    """
    rng = _rng(semilla)
    p = len(coeficientes)
    x = np.zeros(n)
    x[: min(p, n)] = rng.normal(0, ruido, min(p, n))
    c = np.asarray(coeficientes, dtype=float)
    for t in range(p, n):
        x[t] = intercepto + c @ x[t - p : t][::-1] + rng.normal(0, ruido)
    return x


def serie_combinada(n: int = 365, pendiente: float = 0.1, ruido: float = 1.0, periodo: float = 30,
                    amplitud: float = 5.0, coeficientes_ar: tuple[float, ...] = (0.5,), semilla=None) -> np.ndarray:
    """Tendencia + estacionalidad + ruido + componente AR."""
    rng = _rng(semilla)
    t = np.arange(n)
    return (t * pendiente
            + np.sin(2 * np.pi * t / periodo) * amplitud
            + rng.standard_normal(n) * ruido
            + serie_ar(n, list(coeficientes_ar), ruido=ruido / 2, semilla=rng))
