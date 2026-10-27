"""Voto ponderado de varios clasificadores (ensamble).

Viene de ``weighted_vote`` (08_optimizacion_y_series). Cambio de seguridad: el
cuaderno cargaba los modelos con ``joblib.load`` (puede ejecutar código si el
archivo no es tuyo); aquí la función recibe los modelos ya creados.
"""

from __future__ import annotations

import numpy as np


def probabilidades_ponderadas(X, modelos: dict, pesos: dict) -> np.ndarray:
    """Suma ``pesos[nombre] * modelo.predict_proba(X)`` de cada modelo que lo permita."""
    total = None
    for nombre, modelo in modelos.items():
        if not hasattr(modelo, "predict_proba"):
            continue
        p = np.asarray(modelo.predict_proba(X), dtype=float) * float(pesos.get(nombre, 0.0))
        total = p if total is None else total + p
    if total is None:
        raise ValueError("Ningún modelo tiene predict_proba")
    return total


def voto_ponderado(X, modelos: dict, pesos: dict) -> np.ndarray:
    """Clase ganadora (índice) por fila, según las probabilidades ponderadas."""
    return np.argmax(probabilidades_ponderadas(X, modelos, pesos), axis=1)
