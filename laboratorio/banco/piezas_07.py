"""Piezas del cuaderno 07 (grafos).

- **anti_gradiente** («SimpleAntiGradientModel»): extractor → capa de inversión de gradiente (GRL) → clasificador.
  La GRL deja pasar los datos igual, pero al aprender invierte el gradiente: el extractor aprende a **empeorar**
  la misma tarea que el clasificador intenta resolver.

  Rival de la industria: la GRL es de Ganin y Lempitsky (2015, DANN). Allí va en una **rama aparte**, delante de un
  clasificador de «dominio» (de qué conjunto viene cada dato), para que los rasgos sirvan igual en los dos dominios;
  nunca en el camino de la tarea principal. El banco no tiene tareas con dominios, así que se mide tal cual está
  en el cuaderno.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class _InvertirGradiente(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambda_ * grad, None


class RedAntiGradiente(nn.Module):
    """Las dos capas ReLU de siempre, la GRL (λ = 1, como en el cuaderno) y la salida."""

    def __init__(self, entradas: int, salidas: int, ocultas: int, lambda_: float = 1.0) -> None:
        super().__init__()
        self.capa1, self.capa2 = nn.Linear(entradas, ocultas), nn.Linear(ocultas, ocultas)
        self.salida = nn.Linear(ocultas, salidas)
        self.lambda_ = lambda_

    def forward(self, x):
        rasgos = F.relu(self.capa2(F.relu(self.capa1(x))))
        return self.salida(_InvertirGradiente.apply(rasgos, self.lambda_))


def piezas_07():
    from .piezas import Pieza

    return [
        Pieza("anti_gradiente", "Inversión de gradiente entre extractor y clasificador (en el camino principal)",
              "07 · GradientReversal / SimpleAntiGradientModel", None,
              red=lambda e, s, o: RedAntiGradiente(e, s, o)),
    ]
