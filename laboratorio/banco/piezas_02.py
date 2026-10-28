"""Más piezas del cuaderno 02 y las activaciones y capas de la industria contra las que compiten.

Industria (referencias):
- **GELU** (BERT, GPT), **SiLU/Swish** (Llama, EfficientNet), **Mish** (YOLOv4), **ELU**, **LeakyReLU**.
- **Residual**: capa con salto (estilo ResNet), la forma estándar de conectar capas hoy.
- **Tanh**: no es de hoy, pero es el control justo para las piezas de memoria, que usan tanh: así se
  separa el efecto de la idea del efecto de la tanh.

Tus piezas de esta tanda (cuaderno 02):
- memoria de errores (celdas 4-6), memoria de gradiente (8-11), LR por neurona (16),
  memoria con atención (18-19), memoria con embeddings (20-22), memoria con atención y embeddings (23-25);
- conexiones denso-cruzadas (32) y características densas de entrada (34): topologías completas.

Las piezas «con memoria» hacen una corrección extra **después** de cada paso del
optimizador (``despues_del_paso``), como en el cuaderno.
"""

from __future__ import annotations

from collections import deque

import torch
import torch.nn.functional as F
from torch import nn

from .piezas import CapaConActivacion


# ---------------------------------------------------------------- Industria
def _act_industria(nombre: str) -> nn.Module:
    return {"tanh": nn.Tanh(), "gelu": nn.GELU(), "silu": nn.SiLU(), "mish": nn.Mish(), "elu": nn.ELU(), "leaky_relu": nn.LeakyReLU(0.01)}[nombre]


class CapaResidual(nn.Module):
    """Lineal + ReLU con salto (ResNet): si entradas = salidas, suma la entrada; si no, la proyecta."""

    def __init__(self, entradas: int, salidas: int) -> None:
        super().__init__()
        self.lineal = nn.Linear(entradas, salidas)
        self.atajo = nn.Identity() if entradas == salidas else nn.Linear(entradas, salidas, bias=False)

    def forward(self, x):
        return F.relu(self.lineal(x)) + self.atajo(x)


def _pesos_iniciales(salidas: int, entradas: int) -> tuple[nn.Parameter, nn.Parameter]:
    """Arreglo: el cuaderno empezaba con ``randn × 0,1`` (pesos muy pequeños). Con tanh la red quedaba
    atascada en la regresión cúbica (60 veces más error que ReLU, con o sin la regla de memoria).
    Se usa el mismo inicio que ``nn.Linear`` (uniforme ±1/√entradas)."""
    capa = nn.Linear(entradas, salidas)
    return nn.Parameter(capa.weight.detach().clone()), nn.Parameter(capa.bias.detach().clone())


# ---------------------------------------------------------------- Celdas 4-6: memoria de errores
class CapaConMemoriaErrores(nn.Module):
    """Recuerda las ``k`` muestras con más error y, tras cada paso, empuja sus activaciones hacia 0.

    Arreglo: el original guardaba la entrada de la capa pero la regla extra no sabía qué error tenía cada
    muestra hasta que se llamaba a mano; aquí se recibe el error de la red en ``despues_del_paso``.
    """

    def __init__(self, in_features, out_features, k=15, lr_extra=0.005):
        super().__init__()
        self.pesos, self.bias = _pesos_iniciales(out_features, in_features)
        self.k, self.lr_extra = k, lr_extra
        self.buffer_fallas: deque = deque(maxlen=k)

    def forward(self, x):
        self.entrada_actual = x.detach()
        return torch.tanh(F.linear(x, self.pesos, self.bias))

    def despues_del_paso(self, error, **_):
        if not hasattr(self, "entrada_actual") or len(error) != len(self.entrada_actual):
            return
        magnitud = error.abs().reshape(len(error), -1).mean(1)
        for i in magnitud.argsort(descending=True)[: self.k].tolist():
            self.buffer_fallas.append(self.entrada_actual[i])
        entradas = torch.stack(list(self.buffer_fallas))
        with torch.no_grad():
            act = torch.tanh(F.linear(entradas, self.pesos, self.bias))
            self.pesos -= self.lr_extra * (act.T @ entradas) / len(entradas)
            self.bias -= self.lr_extra * act.mean(0)


# ---------------------------------------------------------------- Celdas 8-11: memoria de gradiente
class NeuronaMemoriaGradiente(nn.Module):
    """Guarda los últimos gradientes y, tras cada paso, da un paso extra con su promedio (como un momentum)."""

    def __init__(self, in_features, out_features, memoria_tamano=10, alfa=0.01):
        super().__init__()
        self.w, self.b = _pesos_iniciales(out_features, in_features)
        self.hist_w: deque = deque(maxlen=memoria_tamano)
        self.hist_b: deque = deque(maxlen=memoria_tamano)
        self.alfa = alfa

    def forward(self, x):
        return torch.tanh(F.linear(x, self.w, self.b))

    def despues_del_paso(self, **_):
        if self.w.grad is None:
            return
        self.hist_w.append(self.w.grad.detach().clone())
        self.hist_b.append(self.b.grad.detach().clone())
        with torch.no_grad():
            self.w -= self.alfa * torch.stack(list(self.hist_w)).mean(0)
            self.b -= self.alfa * torch.stack(list(self.hist_b)).mean(0)


# ---------------------------------------------------------------- Celda 16: LR por neurona
class CapaLRporNeurona(nn.Module):
    """Lineal + tanh; tras el paso del optimizador, un ajuste extra con tasa por peso al estilo RMSprop."""

    def __init__(self, in_features, out_features, base_lr_ajuste=0.01, eps=1e-8):
        super().__init__()
        self.pesos, self.bias = _pesos_iniciales(out_features, in_features)
        self.base_lr_ajuste, self.eps = base_lr_ajuste, eps
        self.register_buffer("G_pesos", torch.zeros(out_features, in_features))
        self.register_buffer("G_bias", torch.zeros(out_features))

    def forward(self, x):
        return torch.tanh(F.linear(x, self.pesos, self.bias))

    def despues_del_paso(self, **_):
        if self.pesos.grad is None:
            return
        with torch.no_grad():
            gp, gb = self.pesos.grad, self.bias.grad
            self.G_pesos.mul_(0.9).addcmul_(gp, gp, value=0.1)
            self.G_bias.mul_(0.9).addcmul_(gb, gb, value=0.1)
            self.pesos -= self.base_lr_ajuste / (self.G_pesos.sqrt() + self.eps) * gp
            self.bias -= self.base_lr_ajuste / (self.G_bias.sqrt() + self.eps) * gb


# ---------------------------------------------------------------- Celdas 18-19: memoria con atención
class NeuronaMemoriaAtencion(nn.Module):
    """Recuerda (entrada promedio, gradiente) de lotes pasados; tras cada paso, aplica una mezcla de los
    gradientes recordados, pesada por el parecido (coseno) entre la entrada actual y las recordadas."""

    def __init__(self, in_features, out_features, memoria_tamano=10, alfa=0.05):
        super().__init__()
        self.w, self.b = _pesos_iniciales(out_features, in_features)
        self.mem_x: deque = deque(maxlen=memoria_tamano)
        self.mem_g: deque = deque(maxlen=memoria_tamano)
        self.alfa = alfa

    def forward(self, x):
        self.entrada_actual = x.detach()
        return torch.tanh(F.linear(x, self.w, self.b))

    def despues_del_paso(self, **_):
        if self.w.grad is None or not hasattr(self, "entrada_actual"):
            return
        actual = self.entrada_actual.mean(0, keepdim=True)
        if self.mem_x:
            sim = F.normalize(actual, dim=1) @ F.normalize(torch.cat(list(self.mem_x)), dim=1).T
            pesos = F.softmax(sim, dim=1).T.unsqueeze(-1)
            with torch.no_grad():
                self.w -= self.alfa * (pesos * torch.stack(list(self.mem_g))).sum(0)
        self.mem_x.append(actual)
        self.mem_g.append(self.w.grad.detach().clone())


# ---------------------------------------------------------------- Celdas 20-25: memoria con embeddings
class CapaMemoriaEmbedding(nn.Module):
    """Adaptación a capa de ``NeuronaConMemoriaEmbedding`` (celda 22) y, con ``atencion=True``, de
    ``NeuronaMemoriaAtencionEmbeddings`` (celda 25).

    La capa recuerda embeddings de la pérdida de pasos anteriores. Al calcular, codifica la activación
    promedio de cada ejemplo, la compara con la memoria (coseno, o atención con Q/K/V) y suma a toda la capa un ajuste
    proporcional al parecido. El original era una neurona de salida de una red de regresión; aquí la idea se
    lleva a una capa oculta para poder compararla con las demás piezas.
    """

    def __init__(self, entradas, salidas, memoria=10, dim=32, atencion=False):
        super().__init__()
        self.fc = nn.Linear(entradas, salidas)
        self.encoder = nn.Sequential(nn.Linear(1, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.register_buffer("memoria", torch.zeros(memoria, dim))
        self.ptr = 0
        self.atencion = atencion
        if atencion:
            self.q, self.k, self.v = (nn.Linear(dim, dim, bias=False) for _ in range(3))
            self.escala = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        h = F.relu(self.fc(x))
        # Con la memoria vacía (ceros) el ajuste vale 0, pero se calcula igual para que todo reciba gradiente.
        # Cada ejemplo consulta la memoria con su propia activación media: la predicción no depende del lote.
        actual = self.encoder(h.mean(1, keepdim=True))  # [ejemplos, dim]
        if self.atencion:
            pesos = F.softmax(self.q(actual) @ self.k(self.memoria).T / self.memoria.size(1) ** 0.5, dim=-1)
            ajuste = self.escala * (actual * (pesos @ self.v(self.memoria))).sum(1, keepdim=True)
        else:
            pesos = F.softmax(F.normalize(actual, dim=-1) @ F.normalize(self.memoria, dim=-1).T, dim=-1)
            ajuste = 0.1 * (actual * (pesos @ self.memoria)).sum(1, keepdim=True)
        return h + ajuste

    def despues_del_paso(self, perdida, **_):
        with torch.no_grad():
            self.memoria[self.ptr] = self.encoder(torch.tensor([[float(perdida)]], device=self.memoria.device)).squeeze(0)
            self.ptr = (self.ptr + 1) % len(self.memoria)


# ---------------------------------------------------------------- Inicializador de pesos: Perlin y Lorenz
class CapaInicioEspecial(nn.Module):
    """Lineal + ReLU igual que la estándar; solo cambian los pesos iniciales (Perlin o Lorenz).

    El punto de partida dentro del ruido o de la trayectoria se elige con el generador de PyTorch, así cada
    semilla del banco da un inicio distinto (y repetible).
    """

    def __init__(self, entradas: int, salidas: int, tipo: str) -> None:
        super().__init__()
        import numpy as np

        from ..inicializadores import pesos_perlin, puntos_lorenz

        self.lineal = nn.Linear(entradas, salidas)
        forma = (salidas, entradas)
        if tipo == "perlin":
            desfase = float(_sorteo(100_000))
            pesos = pesos_perlin(forma, desfase=desfase, semilla=_sorteo(2**31))
        else:
            n = salidas * entradas
            trayectoria = _trayectoria_lorenz(puntos_lorenz)
            inicio = _sorteo(len(trayectoria) - n)
            tramo = trayectoria[inicio : inicio + n]
            tramo = (tramo - tramo.mean()) / (tramo.std() or 1.0)
            pesos = (tramo * np.sqrt(2.0 / entradas)).reshape(forma)
        with torch.no_grad():
            self.lineal.weight.copy_(torch.from_numpy(pesos).float())

    def forward(self, x):
        return F.relu(self.lineal(x))


_LORENZ = None


def _sorteo(tope: int) -> int:
    """Número al azar que depende de la semilla, sin mover el generador global: así las demás capas reciben
    exactamente los mismos pesos iniciales que en la red estándar y solo cambia lo que se quiere medir."""
    with torch.random.fork_rng(devices=[]):
        return int(torch.randint(0, tope, ()).item())


def _trayectoria_lorenz(puntos_lorenz):
    """Una sola trayectoria (x, y, z aplanados, sin el transitorio inicial), calculada una vez."""
    global _LORENZ
    if _LORENZ is None:
        _LORENZ = puntos_lorenz(201_000)[1000:].T.ravel()
    return _LORENZ


# ---------------------------------------------------------------- Celdas 32 y 34: topologías densas
class RedDensoCruzada(nn.Module):
    """Celda 32: cada capa recibe la concatenación de todas las anteriores (como DenseNet)."""

    def __init__(self, entradas: int, salidas: int, ocultas: int = 32, con_entrada: bool = False) -> None:
        super().__init__()
        e = entradas if con_entrada else 0
        self.con_entrada = con_entrada
        # Se llaman capa1/capa2 para que las mediciones internas las encuentren igual que en la red común.
        self.capa1 = nn.Linear(entradas, ocultas)
        self.capa2 = nn.Linear(ocultas + e, ocultas)
        self.capa3 = nn.Linear(2 * ocultas + e, ocultas)
        self.salida = nn.Linear(3 * ocultas + e, salidas)

    def forward(self, x):
        extra = [x] if self.con_entrada else []
        h1 = F.relu(self.capa1(x))
        h2 = F.relu(self.capa2(torch.cat([h1, *extra], 1)))
        h3 = F.relu(self.capa3(torch.cat([h1, h2, *extra], 1)))
        return self.salida(torch.cat([h1, h2, h3, *extra], 1))


def piezas_02():
    """Las piezas de este módulo, con el formato de ``piezas.Pieza``."""
    from .piezas import Pieza

    industria = [
        Pieza(n, f"{t} (industria)", "industria", lambda e, s, n=n: CapaConActivacion(e, s, _act_industria(n)),
              lambda c, n=n: _act_industria(n), industria=True)
        for n, t in [("gelu", "GELU: BERT, GPT"), ("silu", "SiLU/Swish: Llama, EfficientNet"),
                     ("mish", "Mish: YOLOv4"), ("elu", "ELU"), ("leaky_relu", "LeakyReLU"),
                     ("tanh", "Tanh clásica: control para tus piezas que usan tanh")]
    ]
    industria.append(Pieza("residual", "Capa con salto, estilo ResNet (industria)", "industria", CapaResidual,
                           industria=True))
    propias = [
        Pieza("memoria_errores", "Recuerda las muestras con más error y las corrige", "02 · celdas 4-6",
              CapaConMemoriaErrores),
        Pieza("memoria_gradiente", "Paso extra con el promedio de gradientes recientes", "02 · celdas 8-11",
              NeuronaMemoriaGradiente),
        Pieza("lr_por_neurona", "Ajuste extra con tasa por peso (RMSprop casero)", "02 · celda 16", CapaLRporNeurona),
        Pieza("memoria_atencion", "Mezcla de gradientes recordados según parecido de la entrada", "02 · celdas 18-19",
              NeuronaMemoriaAtencion),
        Pieza("memoria_embedding", "Memoria de pérdidas como embeddings que ajusta la capa", "02 · celdas 20-22",
              CapaMemoriaEmbedding),
        Pieza("memoria_atencion_embedding", "Igual, con atención Q/K/V sobre la memoria", "02 · celdas 23-25",
              lambda e, s: CapaMemoriaEmbedding(e, s, atencion=True)),
        Pieza("inicio_perlin", "Pesos iniciales con ruido Perlin (escala He)", "02 · inicializador de pesos",
              lambda e, s: CapaInicioEspecial(e, s, "perlin")),
        Pieza("inicio_lorenz", "Pesos iniciales del atractor de Lorenz (escala He)", "02 · inicializador de pesos",
              lambda e, s: CapaInicioEspecial(e, s, "lorenz")),
        Pieza("denso_cruzada", "Cada capa recibe todas las anteriores (topología completa)", "02 · celda 32", None,
              red=lambda e, s, o: RedDensoCruzada(e, s, o)),
        Pieza("densa_entrada", "Igual, y además cada capa recibe la entrada original", "02 · celda 34", None,
              red=lambda e, s, o: RedDensoCruzada(e, s, o, con_entrada=True)),
    ]
    return industria + propias
