"""Piezas del cuaderno 04 (visión): decorrelación y el «papelito preguntón».

El cuaderno estaba en TensorFlow/Keras; aquí se pasó a PyTorch con el mismo cálculo.

- **decorrelacion** (AODC-NN): castiga que las neuronas de una misma capa se parezcan entre sí (covarianza fuera
  de la diagonal). Rival de la industria: es la misma idea que *DeCov* (Cogswell et al., 2016) y que la parte de
  decorrelación de *Barlow Twins*. La versión con convoluciones está en ``arquitecturas.py`` (``cnn_decorrelada``).
- **papelito_pregunton** (ODFE-NN, RDCR-NN): dos capas que «se preguntan»: H1 extrae, H2 ve la entrada y a H1,
  H1 se reajusta viendo a H2 y H2 se reajusta otra vez; una pérdida castiga que H1 y H2 se parezcan (coseno²).
- **papelito_covarianza** (NIDI-NN): igual, pero castiga la covarianza cruzada entre H1 y H2.
- **fusion_reciproca** (RCF-NN): la misma interacción sin pérdida de decorrelación.
- La celda recurrente RCF (H1 y H2 acoplados en cada paso de una secuencia) está en ``arquitecturas.py``
  (``rcf_recurrente``), para series.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def covarianza_fuera_diagonal(h: torch.Tensor) -> torch.Tensor:
    """Suma de cuadrados de la covarianza entre neuronas distintas (h: [ejemplos, neuronas])."""
    if h.shape[0] < 2 or h.shape[1] < 2:
        return h.new_zeros(())
    c = h - h.mean(0, keepdim=True)
    cov = c.T @ c / (h.shape[0] - 1)
    return (cov - torch.diag(torch.diag(cov))).pow(2).sum()


class CapaDecorrelada(nn.Module):
    """Lineal + ReLU con pérdida de decorrelación entre sus neuronas (AODC-NN llevada a una capa densa).

    Arreglo: en las convoluciones del cuaderno la covarianza se calcula sobre los píxeles de cada imagen; en una
    capa densa no hay píxeles, así que se calcula sobre los ejemplos del lote (como DeCov).
    """

    def __init__(self, entradas: int, salidas: int, lambda_: float = 1e-3) -> None:
        super().__init__()
        self.lineal, self.lambda_ = nn.Linear(entradas, salidas), lambda_

    def forward(self, x):
        h = F.relu(self.lineal(x))
        if self.training:
            self.activacion = h
        return h

    def perdida_extra(self, salida, objetivo):
        h = getattr(self, "activacion", None)
        return self.lambda_ * covarianza_fuera_diagonal(h) if h is not None else salida.new_zeros(())


class _Extractor(nn.Module):
    """H1 o H2 del cuaderno: procesa la entrada y, si hay, el contexto de la otra capa; luego combina."""

    def __init__(self, entradas: int, otra: int, unidades: int) -> None:
        super().__init__()
        self.de_entrada = nn.Linear(entradas, unidades)
        self.de_otra = nn.Linear(otra, unidades)
        self.combinar = nn.Linear(2 * unidades, unidades)

    def forward(self, x, contexto=None):
        a = F.relu(self.de_entrada(x))
        if contexto is None:
            return a  # como ExtractorCapaUno del cuaderno: sin contexto devuelve lo extraído de la entrada
        return F.relu(self.combinar(torch.cat([a, F.relu(self.de_otra(contexto))], 1)))


class RedPapelito(nn.Module):
    """El «papelito preguntón»: dos rondas de pregunta y ajuste entre H1 y H2, y salida con las dos.

    ``perdida``: ``"coseno"`` (ODFE/RDCR: media de coseno² entre H1 y H2), ``"covarianza"`` (NIDI: suma de
    cuadrados de la covarianza cruzada) o ``None`` (RCF: sin decorrelación).
    """

    def __init__(self, entradas: int, salidas: int, ocultas: int, perdida: str | None = "coseno",
                 lambda_: float = 0.1) -> None:
        super().__init__()
        self.capa1 = _Extractor(entradas, ocultas, ocultas)  # H1 (nombres capa1/capa2 para las mediciones)
        self.capa2 = _Extractor(entradas, ocultas, ocultas)  # H2: su «contexto» es H1
        self.salida = nn.Linear(2 * ocultas, salidas)
        self.perdida, self.lambda_ = perdida, lambda_

    def forward(self, x):
        h1 = self.capa1(x)
        h2 = self.capa2(x, h1)
        h1 = self.capa1(x, h2)  # H1 se reajusta viendo lo que H2 ya tiene
        h2 = self.capa2(x, h1)  # y H2 vuelve a mirar a H1
        if self.training:
            self.h1, self.h2 = h1, h2
        return self.salida(torch.cat([h1, h2], 1))

    def perdida_extra(self, salida, objetivo):
        if self.perdida is None or not hasattr(self, "h1"):
            return salida.new_zeros(())
        if self.perdida == "coseno":
            return self.lambda_ * F.cosine_similarity(self.h1, self.h2, dim=1).pow(2).mean()
        c1 = self.h1 - self.h1.mean(0)
        c2 = self.h2 - self.h2.mean(0)
        return self.lambda_ * (c1.T @ c2 / len(c1)).pow(2).sum()


def piezas_04():
    from .piezas import Pieza

    return [
        Pieza("decorrelacion", "Castiga que las neuronas de la capa se parezcan (AODC; como DeCov)",
              "04 · AODC-NN", CapaDecorrelada),
        Pieza("papelito_pregunton", "H1 y H2 se preguntan dos veces y se castiga que se parezcan (coseno)",
              "04 · ODFE-NN y RDCR-NN", None, red=lambda e, s, o: RedPapelito(e, s, o, "coseno")),
        Pieza("papelito_covarianza", "Igual, castigando la covarianza cruzada entre H1 y H2", "04 · NIDI-NN", None,
              red=lambda e, s, o: RedPapelito(e, s, o, "covarianza")),
        Pieza("fusion_reciproca", "La misma interacción H1 ↔ H2, sin castigo de parecido", "04 · RCF-NN", None,
              red=lambda e, s, o: RedPapelito(e, s, o, None)),
    ]
