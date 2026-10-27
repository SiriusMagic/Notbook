"""Ayudas para entrenar: bajar la tasa de aprendizaje cuando el modelo se estanca.

Viene de «Programador dinámico de tasa de aprendizaje» (08_optimizacion_y_series).
Funciona con cualquier librería: se le pasa la métrica de cada época y devuelve
la tasa de aprendizaje a usar. (En PyTorch existe ``ReduceLROnPlateau``, que hace
lo mismo; este sirve para redes hechas a mano con NumPy.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ProgramadorLR:
    """Reduce la tasa de aprendizaje si la métrica no mejora en ``paciencia`` épocas.

    >>> p = ProgramadorLR(lr=0.1, paciencia=2, factor=0.5, mayor_es_mejor=False)
    >>> [p.paso(e) for e in (1.0, 0.9, 0.95, 0.97)]
    [0.1, 0.1, 0.1, 0.05]
    """

    lr: float
    paciencia: int = 5
    factor: float = 0.1
    lr_minimo: float = 1e-6
    mayor_es_mejor: bool = True
    mejor: float = field(init=False)
    epocas_sin_mejora: int = field(default=0, init=False)
    mejoro: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not 0 < self.factor < 1:
            raise ValueError("factor debe estar entre 0 y 1")
        if self.paciencia < 1:
            raise ValueError("paciencia debe ser al menos 1")
        self.mejor = -math.inf if self.mayor_es_mejor else math.inf

    def paso(self, metrica: float) -> float:
        """Registra la métrica de esta época y devuelve la tasa de aprendizaje siguiente."""
        self.mejoro = metrica > self.mejor if self.mayor_es_mejor else metrica < self.mejor
        if self.mejoro:
            self.mejor = metrica
            self.epocas_sin_mejora = 0
        else:
            self.epocas_sin_mejora += 1
        if self.epocas_sin_mejora >= self.paciencia:
            self.lr = max(self.lr * self.factor, self.lr_minimo)
            self.epocas_sin_mejora = 0
        return self.lr
