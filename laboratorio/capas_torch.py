"""Capa de inversión de gradiente (Gradient Reversal Layer, GRL) para PyTorch.

Viene de «Inversión de gradiente» (07_grafos_y_generativos). Hacia adelante
deja pasar los datos igual; hacia atrás multiplica el gradiente por ``-lambda``.
Se usa en adaptación de dominio (DANN): la red aprende rasgos que NO permiten
distinguir de qué dominio vienen los datos.

Arreglo: el cuaderno guardaba lambda como tensor y devolvía ``x.view_as(x)``;
aquí lambda es un número y se devuelve una copia, que es lo recomendado.
"""

from __future__ import annotations

import torch
from torch import nn


class _InvertirGradiente(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.clone()

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return -ctx.lambd * grad, None


def invertir_gradiente(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _InvertirGradiente.apply(x, float(lambd))


class CapaInversionGradiente(nn.Module):
    def __init__(self, lambd: float = 1.0) -> None:
        super().__init__()
        self.lambd = float(lambd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return invertir_gradiente(x, self.lambd)

    def extra_repr(self) -> str:
        return f"lambd={self.lambd}"
