"""Piezas de investigación del cuaderno 02 (y la activación seno del 04).

Cada pieza es una capa oculta «entradas → salidas» ya con su activación. Todas
se montan en la misma red (dos capas ocultas + una lineal de salida), así lo
único que cambia entre corridas es la pieza.

Se copiaron del cuaderno casi tal cual; los únicos cambios son para que
funcionen dentro del banco y están marcados con «Arreglo:».
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------- Base de comparación
class CapaEstandar(nn.Module):
    """Lineal + ReLU: la red de referencia («RedEstandar» de los cuadernos)."""

    def __init__(self, entradas: int, salidas: int) -> None:
        super().__init__()
        self.lineal = nn.Linear(entradas, salidas)

    def forward(self, x):
        return F.relu(self.lineal(x))


class CapaConActivacion(nn.Module):
    def __init__(self, entradas: int, salidas: int, activacion: nn.Module) -> None:
        super().__init__()
        self.lineal = nn.Linear(entradas, salidas)
        self.activacion = activacion

    def forward(self, x):
        return self.activacion(self.lineal(x))


# ---------------------------------------------------------------- Celda 12
class CapaUmbralDinamico(nn.Module):
    """El bias se corrige después de cada paso según la activación media y el error."""

    def __init__(self, in_features, out_features, alpha=0.05, beta=0.01):
        super().__init__()
        self.pesos = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.alpha, self.beta = alpha, beta
        self.register_buffer("media_activacion", torch.zeros(out_features))
        self.register_buffer("media_error", torch.zeros(out_features))

    def forward(self, x):
        self.activaciones = torch.tanh(F.linear(x, self.pesos, self.bias))
        return self.activaciones

    def despues_del_paso(self, error, **_):
        # Arreglo: el cuaderno llamaba ajustar_umbral() a mano; aquí lo llama el banco.
        act_mean = self.activaciones.mean(dim=0).detach()
        self.media_activacion = (1 - self.alpha) * self.media_activacion + self.alpha * act_mean
        error_abs = error.abs().mean().detach()
        self.media_error = (1 - self.alpha) * self.media_error + self.alpha * error_abs
        with torch.no_grad():
            self.bias -= self.beta * (self.media_activacion + error_abs)


# ---------------------------------------------------------------- Celda 29
class NeuronaIntegracionNoLineal(nn.Module):
    """En vez de sumar w·x, un mini-MLP integra las señales ponderadas."""

    def __init__(self, in_features, integration_mlp_hidden_size=8):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1, in_features) * 0.1)
        self.b = nn.Parameter(torch.zeros(1))
        self.integrator_mlp = nn.Sequential(
            nn.Linear(in_features, integration_mlp_hidden_size), nn.ReLU(), nn.Linear(integration_mlp_hidden_size, 1)
        )

    def forward(self, x):
        return torch.tanh(self.integrator_mlp(x * self.w) + self.b)


class CapaIntegracionNoLineal(nn.Module):
    # Arreglo: el cuaderno definía una sola neurona; aquí se apilan «salidas» neuronas.
    def __init__(self, entradas, salidas):
        super().__init__()
        self.neuronas = nn.ModuleList(NeuronaIntegracionNoLineal(entradas) for _ in range(salidas))

    def forward(self, x):
        return torch.cat([n(x) for n in self.neuronas], dim=1)


# ---------------------------------------------------------------- Celda 36
class AdaptiveActivationNeuron(nn.Module):
    """Cada neurona aprende cuánto mezclar ReLU y sigmoide (puerta con softmax)."""

    def __init__(self, input_features, output_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_features, output_features) * 0.1)
        self.bias = nn.Parameter(torch.randn(output_features) * 0.1)
        self.gate_weights = nn.Parameter(torch.randn(output_features, 2) * 0.1)

    def forward(self, x):
        z = x @ self.weight + self.bias
        mezcla = torch.stack([F.relu(z), torch.sigmoid(z)], dim=-1)
        return (mezcla * F.softmax(self.gate_weights, dim=-1).unsqueeze(0)).sum(-1)


# ---------------------------------------------------------------- Celda 39
class IDANeuron(nn.Module):
    """Activación con pendiente, desplazamiento y forma aprendibles (ReLU + tanh)."""

    def __init__(self, input_features, output_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_features, output_features) * 0.1)
        self.bias = nn.Parameter(torch.randn(output_features) * 0.1)
        self.slope_param = nn.Parameter(torch.ones(output_features) * 0.1)
        self.offset_param = nn.Parameter(torch.zeros(output_features))
        self.shape_param = nn.Parameter(torch.zeros(output_features))

    def forward(self, x):
        z = x @ self.weight + self.bias
        relu_like = F.relu(z - self.offset_param) * torch.sigmoid(self.slope_param)
        tanh_like = torch.tanh(z) * torch.sigmoid(self.shape_param)
        return relu_like + tanh_like


# ---------------------------------------------------------------- Celda 40
ACTIVACIONES_SMA = {"relu": F.relu, "sigmoid": torch.sigmoid, "tanh": torch.tanh, "leaky_relu": F.leaky_relu}


class SMANeuron(nn.Module):
    """Cada neurona «muta» su activación si su gradiente es casi nulo o muy alto."""

    def __init__(self, input_features, output_features, initial_activation="relu", umbral_mutacion=0.01, enfriamiento=50):
        super().__init__()
        self.output_features = output_features
        self.weight = nn.Parameter(torch.randn(input_features, output_features) * 0.1)
        self.bias = nn.Parameter(torch.randn(output_features) * 0.1)
        self.current_activations = [initial_activation] * output_features
        self._cooldown = [0] * output_features
        self.umbral_mutacion, self.enfriamiento = umbral_mutacion, enfriamiento
        self.mutaciones = 0

    def forward(self, x):
        z = x @ self.weight + self.bias
        return torch.stack([ACTIVACIONES_SMA[a](z[:, i]) for i, a in enumerate(self.current_activations)], dim=1)

    def despues_del_paso(self, perdida, epoca, **_):
        # Arreglo: la norma del gradiente se lee de weight.grad (el cuaderno usaba un hook
        # que guardaba floats por neurona) y el azar usa torch para poder repetir corridas.
        if self.weight.grad is None:
            return
        normas = self.weight.grad.norm(dim=0)
        nombres = list(ACTIVACIONES_SMA)
        for i in range(self.output_features):
            if self._cooldown[i] > 0:
                self._cooldown[i] -= 1
                continue
            g = float(normas[i])
            inanicion = g < 0.001 and perdida > self.umbral_mutacion and epoca > 100
            estres = g > 1.0 and perdida > self.umbral_mutacion * 2 and epoca > 100
            if inanicion or estres:
                otras = [n for n in nombres if n != self.current_activations[i]]
                self.current_activations[i] = otras[int(torch.randint(len(otras), (1,)))]
                self._cooldown[i] = self.enfriamiento
                self.mutaciones += 1


# ---------------------------------------------------------------- Celda 76
class RMBLayer(nn.Module):
    """El bias depende de una memoria (media móvil) de las salidas anteriores de la capa."""

    def __init__(self, input_features, output_features, memory_decay=0.9):
        super().__init__()
        self.memory_decay = memory_decay
        self.weight = nn.Parameter(torch.randn(input_features, output_features) * 0.1)
        self.base_bias = nn.Parameter(torch.randn(output_features) * 0.1)
        self.register_buffer("residual_memory", torch.zeros(output_features))
        self.memory_transform = nn.Linear(output_features, output_features)
        nn.init.zeros_(self.memory_transform.weight)
        nn.init.zeros_(self.memory_transform.bias)

    def forward(self, x):
        sesgo = self.base_bias + self.memory_transform(self.residual_memory.unsqueeze(0)).squeeze(0)
        salida = x @ self.weight + sesgo.unsqueeze(0)
        # Arreglo: la memoria solo se actualiza al entrenar; antes también cambiaba al evaluar.
        if self.training:
            # Se asigna un tensor nuevo (no se modifica en el lugar): el viejo aún se usa para el gradiente.
            self.residual_memory = (self.memory_decay * self.residual_memory
                                    + (1 - self.memory_decay) * salida.detach().mean(0))
        return F.relu(salida)


# ---------------------------------------------------------------- Celda 80
class HMNLayer(nn.Module):
    """Cada neurona tiene su propia mini-LSTM que recuerda entre pasos."""

    def __init__(self, input_features, output_features):
        super().__init__()
        self.output_features = output_features
        self.weight = nn.Parameter(torch.randn(input_features, output_features) * 0.1)
        self.bias = nn.Parameter(torch.randn(output_features) * 0.1)
        self.lstm_cells = nn.ModuleList(nn.LSTMCell(1, 1) for _ in range(output_features))
        self.h_states: torch.Tensor | None = None
        self.c_states: torch.Tensor | None = None

    def forward(self, x):
        z = x @ self.weight + self.bias
        n = x.size(0)
        # Arreglo: estados como atributos simples (no buffers que cambian de forma) y sin print.
        if self.h_states is None or self.h_states.shape != (n, self.output_features):
            self.h_states = z.new_zeros(n, self.output_features)
            self.c_states = z.new_zeros(n, self.output_features)
        hs, cs = [], []
        for i, celda in enumerate(self.lstm_cells):
            h, c = celda(z[:, i : i + 1], (self.h_states[:, i : i + 1], self.c_states[:, i : i + 1]))
            hs.append(h)
            cs.append(c)
        salida = torch.cat(hs, dim=1)
        self.h_states, self.c_states = salida.detach(), torch.cat(cs, dim=1).detach()
        return salida


# ---------------------------------------------------------------- Cuaderno 04
class ActivacionSeno(nn.Module):
    """sin(w0·x), como en el ajuste de imágenes del cuaderno 04 (allí con w0 = 1, en Keras)."""

    def __init__(self, w0: float = 1.0) -> None:
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


# ---------------------------------------------------------------- Activaciones para CNN, texto y series
# Las piezas que son «una activación» también se pueden poner dentro de una CNN,
# en lugar de ReLU. Sus parámetros van por canal (como PReLU en PyTorch).
def _por_canal(p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Da forma (1, C, 1, …) a un parámetro de C valores para que encaje con x (N, C, …) o (N, C)."""
    return p.view(1, -1, *([1] * (x.dim() - 2)))


class ActivacionIDA(nn.Module):
    """La activación de IDANeuron (celda 39) sin su capa lineal, para usarla en cualquier red."""

    def __init__(self, canales: int) -> None:
        super().__init__()
        self.slope_param = nn.Parameter(torch.ones(canales) * 0.1)
        self.offset_param = nn.Parameter(torch.zeros(canales))
        self.shape_param = nn.Parameter(torch.zeros(canales))

    def forward(self, z):
        return (F.relu(z - _por_canal(self.offset_param, z)) * torch.sigmoid(_por_canal(self.slope_param, z))
                + torch.tanh(z) * torch.sigmoid(_por_canal(self.shape_param, z)))


class ActivacionAdaptativa(nn.Module):
    """La mezcla ReLU/sigmoide de AdaptiveActivationNeuron (celda 36), por canal."""

    def __init__(self, canales: int) -> None:
        super().__init__()
        self.gate_weights = nn.Parameter(torch.randn(canales, 2) * 0.1)

    def forward(self, z):
        g = F.softmax(self.gate_weights, dim=-1)
        return _por_canal(g[:, 0], z) * F.relu(z) + _por_canal(g[:, 1], z) * torch.sigmoid(z)


# ---------------------------------------------------------------- Registro
@dataclass(frozen=True)
class Pieza:
    nombre: str
    descripcion: str
    origen: str
    capa: Callable[[int, int], nn.Module] | None  # (entradas, salidas) → capa oculta con activación
    # Si la pieza es una activación: cómo crearla para C canales (se usa dentro de CNN, texto y series).
    # Si es None, en esas arquitecturas la pieza solo va en la parte final (la «cabeza»).
    activacion: Callable[[int], nn.Module] | None = None
    industria: bool = False  # True = tecnología de hoy (GELU, SiLU…): rival, no idea tuya
    # Si la idea es una forma de conectar las capas (no una capa): (entradas, salidas, ocultas) → red completa.
    red: Callable[[int, int, int], nn.Module] | None = None


PIEZAS: dict[str, Pieza] = {p.nombre: p for p in [
    Pieza("estandar", "Lineal + ReLU (referencia)", "RedEstandar", CapaEstandar, lambda c: nn.ReLU()),
    Pieza("prelu_adaptativa", "PReLU con pendiente aprendible por neurona", "02 · celda 27",
          lambda e, s: CapaConActivacion(e, s, nn.PReLU(s)), lambda c: nn.PReLU(c)),
    Pieza("integracion_no_lineal", "Un mini-MLP reemplaza la suma ponderada", "02 · celda 29", CapaIntegracionNoLineal),
    Pieza("activacion_adaptativa", "Mezcla aprendida de ReLU y sigmoide", "02 · celda 36", AdaptiveActivationNeuron,
          ActivacionAdaptativa),
    Pieza("ida", "Activación con forma aprendible (IDA)", "02 · celda 39", IDANeuron, ActivacionIDA),
    Pieza("sma", "Activación que muta según el gradiente (SMA)", "02 · celda 40", SMANeuron),
    Pieza("umbral_dinamico", "Bias corregido por activación y error", "02 · celda 12", CapaUmbralDinamico),
    Pieza("rmb", "Sesgo con memoria residual (RMB)", "02 · celda 76", RMBLayer),
    Pieza("hmn", "Neuronas con memoria LSTM propia (HMN)", "02 · celda 80", HMNLayer),
    Pieza("seno", "Activación seno", "04 · ajuste de imagen", lambda e, s: CapaConActivacion(e, s, ActivacionSeno()),
          lambda c: ActivacionSeno()),
]}


class Red(nn.Module):
    """Red común: pieza(entradas→ocultas) → pieza(ocultas→ocultas) → Lineal(ocultas→salidas)."""

    def __init__(self, pieza: Pieza, entradas: int, salidas: int, ocultas: int = 32) -> None:
        super().__init__()
        if pieza.red is not None:  # la pieza es la forma de conectar: ella arma toda la red
            self.topologia = pieza.red(entradas, salidas, ocultas)
            return
        self.capa1 = pieza.capa(entradas, ocultas)
        self.capa2 = pieza.capa(ocultas, ocultas)
        self.salida = nn.Linear(ocultas, salidas)

    def forward(self, x):
        if hasattr(self, "topologia"):
            return self.topologia(x)
        return self.salida(self.capa2(self.capa1(x)))

    def despues_del_paso(self, **contexto) -> None:
        """Le avisa a las piezas que tienen reglas propias (umbral dinámico, mutaciones, memorias)."""
        for m in (getattr(self, "capa1", None), getattr(self, "capa2", None), getattr(self, "topologia", None)):
            if hasattr(m, "despues_del_paso"):
                m.despues_del_paso(**contexto)

    def perdida_extra(self, salida, objetivo):
        """Pérdida adicional de las piezas que la tienen (p. ej. el extractor de CDM); si no, 0."""
        total = salida.new_zeros(())
        for m in (getattr(self, "capa1", None), getattr(self, "capa2", None), getattr(self, "topologia", None)):
            if hasattr(m, "perdida_extra"):
                total = total + m.perdida_extra(salida, objetivo)
        return total


# Tanda 2 del cuaderno 02 y sus rivales de la industria (en otro archivo para no alargar este).
from .piezas_02 import piezas_02  # noqa: E402

from .piezas_02b import piezas_02b  # noqa: E402
from .piezas_02c import piezas_02c  # noqa: E402

PIEZAS.update({p.nombre: p for p in piezas_02()})
PIEZAS.update({p.nombre: p for p in piezas_02b()})
PIEZAS.update({p.nombre: p for p in piezas_02c()})


def contar_parametros(modelo: nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)
