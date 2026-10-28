"""Cuaderno 02, última parte: neurona dendrítica, synaptic tagging, estado interno y ruteo adaptativo (RARC).

La CNN con entrada densa del mismo cuaderno está en ``arquitecturas.py`` (``cnn_densa_entrada``), porque es una
arquitectura de imágenes completa.

Rival de la industria del ruteo: la **mezcla de expertos** (Mixtral, Switch Transformer). RARC es justamente
una mezcla suave de dos expertos, así que se compara contra la red estándar del mismo tamaño.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .piezas_02 import _pesos_iniciales


# ---------------------------------------------------------------- Neurona dendrítica (NumPy, celdas 58-64)
class CapaDendritica(nn.Module):
    """Cada neurona tiene dos dendritas: la primera mitad de las entradas pasa por ReLU, la segunda por una
    exponencial suave (exp(z/10)). El soma suma las dos ramas y el sesgo y aplica sigmoide.

    Arreglos:
    - El cuaderno era una sola neurona escrita a mano en NumPy (gradiente calculado a mano, un ejemplo a la
      vez). Aquí es una capa de PyTorch con el mismo cálculo; el gradiente lo saca autograd.
    - Los pesos empezaban en ``rand × 0,1`` (todos positivos y pequeños); se usa el inicio de ``nn.Linear``,
      como en las capas con memoria, donde el inicio pequeño las dejaba atascadas.
    - Con 1 entrada no hay dos mitades: la segunda dendrita recibe la misma entrada.
    """

    def __init__(self, entradas: int, salidas: int, escala: float = 10.0) -> None:
        super().__init__()
        self.corte = max(1, entradas // 2)
        resto = entradas - self.corte if entradas > 1 else 1
        self.dendrita1 = nn.Linear(self.corte, salidas, bias=False)
        self.dendrita2 = nn.Linear(resto, salidas, bias=False)
        self.sesgo = nn.Parameter(torch.zeros(salidas))
        self.escala = escala

    def forward(self, x):
        x1, x2 = (x[:, : self.corte], x[:, self.corte :]) if x.shape[1] > 1 else (x, x)
        rama1 = F.relu(self.dendrita1(x1))
        rama2 = torch.exp((self.dendrita2(x2) / self.escala).clamp(-50, 50))
        return torch.sigmoid(rama1 + rama2 + self.sesgo)


# ---------------------------------------------------------------- Synaptic tagging (celda 66, parte PyTorch)
class CapaSynapticTagging(nn.Module):
    """Lineal + tanh que marca («etiqueta») los pesos con gradiente fuerte; cada ``periodo`` pasos refuerza
    los pesos etiquetados cuyo gradiente sigue en la misma dirección (consolidación, como en la memoria).

    Arreglos:
    - En el cuaderno las etiquetas y la consolidación se aplicaban antes de ``optimizer.step()``; aquí justo
      después, con el mismo gradiente (el efecto es el mismo: una corrección extra que se suma).
    - Inicio de pesos de ``nn.Linear`` en lugar de ``randn × 0,1``.
    """

    def __init__(self, entradas: int, salidas: int, umbral=0.01, decaimiento=0.1, fuerza=0.005, periodo=10) -> None:
        super().__init__()
        self.pesos, self.bias = _pesos_iniciales(salidas, entradas)
        self.umbral, self.decaimiento, self.fuerza, self.periodo = umbral, decaimiento, fuerza, periodo
        self.register_buffer("etiquetas", torch.zeros(salidas, entradas))
        self.register_buffer("direcciones", torch.zeros(salidas, entradas))
        self.pasos = 0
        self.consolidados = 0  # cuántos pesos se reforzaron en total (para ver si la regla actúa)

    def forward(self, x):
        return torch.tanh(F.linear(x, self.pesos, self.bias))

    def despues_del_paso(self, **_):
        if self.pesos.grad is None:
            return
        g = self.pesos.grad
        with torch.no_grad():
            self.etiquetas.mul_(1 - self.decaimiento)
            fuerte = g.abs() > self.umbral
            self.etiquetas[fuerte] = 1.0
            self.direcciones[fuerte] = torch.sign(g[fuerte])
            self.pasos += 1
            if self.pasos % self.periodo == 0:
                mascara = (self.etiquetas > 0.1) & (torch.sign(g) == self.direcciones)
                self.pesos[mascara] -= self.fuerza * g[mascara]
                self.consolidados += int(mascara.sum())


# ---------------------------------------------------------------- Estado interno (celdas 66 y 72, TensorFlow)
class CapaEstadoInterno(nn.Module):
    """``NeuronaEstadoInterno``: tanh(W·x + U·s₀ + b), con un estado inicial s₀ aprendido.

    Se pasó de TensorFlow a PyTorch. Ojo: el estado se crea de nuevo en cada llamada y se usa una sola vez, así
    que U·s₀ es un vector fijo: la capa equivale a lineal + tanh con un sesgo extra (no recuerda nada entre
    ejemplos). Se mide igual, y el control justo es la pieza ``tanh``.
    """

    def __init__(self, entradas: int, salidas: int) -> None:
        super().__init__()
        self.w = nn.Linear(entradas, salidas)
        self.u = nn.Linear(salidas, salidas, bias=False)
        nn.init.orthogonal_(self.u.weight)
        self.estado_inicial = nn.Parameter(torch.zeros(salidas))

    def forward(self, x):
        return torch.tanh(self.w(x) + self.u(self.estado_inicial))


# ---------------------------------------------------------------- Ruteo adaptativo RARC (celda 74, TensorFlow)
class RedRuteoAdaptativo(nn.Module):
    """RARC-NN: una red «compuerta» reparte cada ejemplo entre dos rutas (softmax):
    - ruta de **éxito**: dos capas ReLU;
    - ruta de **reevaluación**: una capa tanh;
    cada ruta va a la salida y se suman con los pesos de la compuerta. Al final, una capa lineal más.

    Pasada de TensorFlow a PyTorch. La salida final del cuaderno tenía softmax; aquí da logits (el banco
    aplica la pérdida adecuada). Tamaños: las dos capas de éxito usan ``ocultas`` y la de reevaluación la mitad.
    """

    def __init__(self, entradas: int, salidas: int, ocultas: int) -> None:
        super().__init__()
        self.compuerta = nn.Linear(entradas, 2)
        self.capa1 = nn.Linear(entradas, ocultas)  # ruta de éxito (nombres capa1/capa2 para las mediciones)
        self.capa2 = nn.Linear(ocultas, ocultas)
        self.reevaluacion = nn.Linear(entradas, max(1, ocultas // 2))
        self.salida_exito = nn.Linear(ocultas, salidas)
        self.salida_reevaluacion = nn.Linear(max(1, ocultas // 2), salidas)
        self.final = nn.Linear(salidas, salidas)

    def forward(self, x):
        pesos = F.softmax(self.compuerta(x), dim=1)
        exito = self.salida_exito(F.relu(self.capa2(F.relu(self.capa1(x)))))
        reeval = self.salida_reevaluacion(torch.tanh(self.reevaluacion(x)))
        return self.final(pesos[:, :1] * exito + pesos[:, 1:] * reeval)


def piezas_02c():
    from .piezas import Pieza

    return [
        Pieza("dendritica", "Dos dendritas por neurona (ReLU y exponencial) y soma sigmoide", "02 · neurona dendrítica",
              CapaDendritica),
        Pieza("synaptic_tagging", "Etiqueta los pesos con gradiente fuerte y los consolida cada 10 pasos",
              "02 · synaptic tagging", CapaSynapticTagging),
        Pieza("estado_interno", "tanh con un estado inicial aprendido (pasada de TensorFlow)", "02 · estado interno",
              CapaEstadoInterno),
        Pieza("ruteo_adaptativo", "Compuerta que reparte entre una ruta de éxito y una de reevaluación (RARC)",
              "02 · ruteo adaptativo", None, red=lambda e, s, o: RedRuteoAdaptativo(e, s, o)),
    ]
