"""Mejoras de las capas más prometedoras (dendrítica y base radial), para intentar superar a Dropout.

- **dendritica_dropout** / **rbf_dropout**: la pieza y, encima, el Dropout de la industria. Si las dos frenan la
  memorización de forma distinta, juntas deberían sumar.
- **dendritica_multi**: cada neurona con 4 dendritas que ven **todas** las entradas (la original solo tenía 2 y cada
  una veía media entrada), cada dendrita con su propia no linealidad (ReLU, exponencial suave, seno, base radial) y
  el soma sigmoide de la original. Así cada neurona combina detectores de formas distintas.
- **rbf_selectiva**: la base radial, pero aprende una escala por columna de la entrada (puede ignorar columnas de
  ruido).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .piezas_02c import CapaDendritica
from .piezas_reparadas import CapaRBF


class ConDropout(nn.Module):
    def __init__(self, capa: nn.Module, p: float = 0.1) -> None:
        super().__init__()
        self.capa, self.dropout = capa, nn.Dropout(p)

    def forward(self, x):
        return self.dropout(self.capa(x))


class CapaDendriticaMulti(nn.Module):
    """soma = sigmoide(ReLU(W₁x) + exp(W₂x/10) + sen(W₃x) + exp(−‖x − c‖²/2σ²) + b)."""

    def __init__(self, entradas: int, salidas: int) -> None:
        super().__init__()
        self.dendritas = nn.Linear(entradas, 3 * salidas, bias=False)
        self.rbf = CapaRBF(entradas, salidas)
        self.sesgo = nn.Parameter(torch.zeros(salidas))
        self.salidas = salidas

    def forward(self, x):
        z1, z2, z3 = self.dendritas(x).split(self.salidas, dim=1)
        suma = F.relu(z1) + torch.exp((z2 / 10).clamp(-50, 50)) + torch.sin(z3) + self.rbf(x) + self.sesgo
        return torch.sigmoid(suma)


class CapaRBFSelectiva(CapaRBF):
    """Base radial que además aprende **cuánto importa cada columna** de la entrada (una escala por columna,
    como la «relevancia automática» de los procesos gaussianos). En datos con columnas de ruido, puede bajar su
    escala a casi 0 y dejar de mirarlas."""

    def __init__(self, entradas: int, salidas: int) -> None:
        super().__init__(entradas, salidas)
        self.log_escala = nn.Parameter(torch.zeros(entradas))

    def forward(self, x):
        escala = torch.exp(self.log_escala)
        if self.training and not bool(self.iniciada):
            super().forward(x)  # inicia centros y ancho con la entrada sin escalar
        xs, cs = x * escala, self.centros * escala
        d2 = torch.cdist(xs, cs).pow(2)
        sigma2 = torch.exp(2 * self.log_sigma).clamp(min=1e-4)
        return torch.exp(-d2 / (2 * sigma2)) + 0.1 * F.relu(self.lineal(x))


def piezas_mejoradas():
    from .piezas import Pieza

    return [
        Pieza("dendritica_dropout", "Dendrítica + Dropout 10 %", "02 · mejorada",
              lambda e, s: ConDropout(CapaDendritica(e, s))),
        Pieza("rbf_dropout", "Base radial entrenable + Dropout 10 %", "03 · mejorada",
              lambda e, s: ConDropout(CapaRBF(e, s))),
        Pieza("rbf_selectiva", "Base radial que aprende qué columnas de la entrada mirar", "03 · mejorada",
              CapaRBFSelectiva),
        Pieza("dendritica_multi", "Dendrítica con 4 dendritas distintas que ven toda la entrada", "02 · mejorada",
              CapaDendriticaMulti),
    ]
