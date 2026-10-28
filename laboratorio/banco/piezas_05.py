"""Piezas del cuaderno 05 (atención).

- **atencion_salida** («AttentiveLinear»): cada neurona reparte su activación hacia la capa siguiente con
  «factores de atención» sigmoide(A). Ojo: A es una matriz fija, no depende de la entrada, así que la capa
  equivale a lineal + ReLU seguida de otra lineal con pesos entre 0 y 1 y sin sesgo. No es atención en el
  sentido de los Transformers (que cambia con cada entrada).
- **atencion_local** (TensorFlow, series): guarda las últimas 5 entradas y las promedia pesando cada una por
  1/error, y concatena ese promedio a la entrada actual.

La «atención suave con RL» del mismo cuaderno está en ``arquitecturas.py`` (``transformer_mascara_rl``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class AtencionSalida(nn.Module):
    """``AttentiveLinear``: relu(W·x + b), y luego cada neurona i envía a_i · sigmoide(A[i, j]) a la neurona j."""

    def __init__(self, entradas: int, salidas: int) -> None:
        super().__init__()
        self.lineal = nn.Linear(entradas, salidas)
        self.atencion = nn.Parameter(torch.empty(salidas, salidas).uniform_(-0.1, 0.1))

    def forward(self, x):
        return F.relu(self.lineal(x)) @ torch.sigmoid(self.atencion)


class AtencionLocal(nn.Module):
    """``CapaAtencionLocal``: [entrada, promedio de las últimas 5 entradas pesadas por 1/error] → lineal + ReLU.

    Arreglos:
    - El cuaderno le pasaba siempre ``loss_actual = 1.0`` (un valor de relleno), así que el peso por error nunca
      actuaba y era un promedio simple. Aquí recibe la pérdida real de cada paso (``despues_del_paso``).
    - El cuaderno iba de a un ejemplo; aquí se guarda el promedio de cada lote.

    Al evaluar, la memoria queda fija: lo que se agrega a cada entrada es un vector constante.
    """

    def __init__(self, entradas: int, salidas: int, memoria: int = 5) -> None:
        super().__init__()
        self.lineal = nn.Linear(2 * entradas, salidas)
        self.register_buffer("memoria", torch.zeros(memoria, entradas))
        self.register_buffer("errores", torch.ones(memoria))

    def forward(self, x):
        if self.training:
            self.entrada_actual = x.detach().mean(0)
        pesos = 1.0 / (self.errores + 1e-6)
        pasado = (self.memoria * (pesos / pesos.sum()).unsqueeze(1)).sum(0)
        return F.relu(self.lineal(torch.cat([x, pasado.expand(len(x), -1)], 1)))

    @torch.no_grad()
    def despues_del_paso(self, perdida=1.0, **_):
        if not hasattr(self, "entrada_actual"):
            return
        self.memoria.copy_(torch.cat([self.memoria[1:], self.entrada_actual.unsqueeze(0)]))
        self.errores.copy_(torch.cat([self.errores[1:], torch.tensor([float(perdida)], device=self.errores.device)]))


def piezas_05():
    from .piezas import Pieza

    return [
        Pieza("atencion_salida", "Cada neurona reparte su salida con factores fijos entre 0 y 1 (AttentiveLinear)",
              "05 · atención a la salida", AtencionSalida),
        Pieza("atencion_local", "Agrega el promedio de entradas pasadas pesado por 1/error", "05 · atención local",
              AtencionLocal),
    ]
