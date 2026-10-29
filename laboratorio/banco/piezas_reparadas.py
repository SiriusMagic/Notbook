"""Piezas «no listas» de RECOMENDACIONES.md, reparadas o con otro enfoque.

Cada una arregla el error de diseño que la hacía perder, sin cambiar la idea de fondo:

- **atencion_dinamica** (AttentiveLinear, 05): los factores de atención ahora dependen de la entrada
  (compuerta sigmoide calculada desde la propia salida, como Squeeze-and-Excitation o GLU).
- **rbf_entrenable** (neuronas KNN, 03): los prototipos y su ancho se aprenden con gradiente, no solo la salida.
- **hmn_sin_memoria** (HMN, 02): la mini-LSTM de cada neurona arranca de cero en cada ejemplo; ya no puede
  memorizar la posición de la fila. Queda como una activación con compuertas aprendida por neurona.
- **rmb_estable** (RMB, 02): pesos iniciados como ``nn.Linear`` y memoria acotada con tanh.
- **resonante_suave** (02): el umbral duro se cambia por una compuerta sigmoide; el umbral y su pendiente se aprenden.
- **hiper_relu_suave** (CCN, 02): la amplificación es un parámetro que empieza en 1 (no ×10 fijo) y está acotada.
- **polinomica_regularizada** (02): términos cuadráticos de rango bajo, entrada acotada con tanh y una
  penalización pequeña sobre ellos para que no memorice.
- **anti_gradiente_dann** (07): la inversión de gradiente va en una rama aparte que adivina el «dominio», como
  DANN. Solo tiene sentido cuando hay dos dominios; se mide con :func:`competir_dann` (dígitos limpios → dígitos
  movidos y con ruido, sin etiquetas del segundo).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class AtencionDinamica(nn.Module):
    """h = relu(W·x + b); compuerta = sigmoide(A·h + c); salida = h · compuerta (cambia con cada entrada)."""

    def __init__(self, entradas: int, salidas: int) -> None:
        super().__init__()
        self.lineal = nn.Linear(entradas, salidas)
        self.compuerta = nn.Linear(salidas, salidas)
        nn.init.zeros_(self.compuerta.weight)
        nn.init.constant_(self.compuerta.bias, 2.0)  # empieza casi abierta: arranca como lineal + ReLU

    def forward(self, x):
        h = F.relu(self.lineal(x))
        return h * torch.sigmoid(self.compuerta(h))


class CapaRBF(nn.Module):
    """Neuronas de base radial: exp(-‖x − c‖² / 2σ²), con centros c y anchos σ aprendidos por gradiente.

    Los centros empiezan en ejemplos reales del primer lote y σ en la distancia típica (como en CapaKNN).
    Se suma una parte lineal pequeña para que la red no dependa solo de las distancias.
    """

    def __init__(self, entradas: int, salidas: int) -> None:
        super().__init__()
        self.centros = nn.Parameter(torch.randn(salidas, entradas))
        self.log_sigma = nn.Parameter(torch.zeros(salidas))
        self.lineal = nn.Linear(entradas, salidas)
        self.register_buffer("iniciada", torch.tensor(False))

    def forward(self, x):
        if self.training and not bool(self.iniciada):
            with torch.no_grad():
                idx = torch.randint(0, len(x), (len(self.centros),), device=x.device)
                self.centros.copy_(x[idx] + 0.01 * torch.randn_like(self.centros))
                d = torch.cdist(x, self.centros)
                self.log_sigma.fill_(math.log(float(d.median().clamp(min=1e-3))))
                self.iniciada.fill_(True)
        d2 = torch.cdist(x, self.centros).pow(2)
        sigma2 = torch.exp(2 * self.log_sigma).clamp(min=1e-4)
        return torch.exp(-d2 / (2 * sigma2)) + 0.1 * F.relu(self.lineal(x))


class CapaHMNSinMemoria(nn.Module):
    """Lineal y, por neurona, una celda LSTM de 1 número que empieza en cero en cada ejemplo.

    Es la misma cuenta que ``nn.LSTMCell(1, 1)`` con estado (0, 0): h = o · tanh(i · g), con
    i, g, o = compuertas de z. Se escribe junta para todas las neuronas (una sola operación).
    """

    def __init__(self, entradas: int, salidas: int) -> None:
        super().__init__()
        self.lineal = nn.Linear(entradas, salidas)
        self.w = nn.Parameter(torch.randn(3, salidas) * 0.5 + torch.tensor([[1.0], [1.0], [1.0]]))
        self.b = nn.Parameter(torch.zeros(3, salidas))

    def forward(self, x):
        z = self.lineal(x).unsqueeze(0) * self.w.unsqueeze(1) + self.b.unsqueeze(1)  # [3, lote, salidas]
        i, g, o = torch.sigmoid(z[0]), torch.tanh(z[1]), torch.sigmoid(z[2])
        return o * torch.tanh(i * g)


class CapaRMBEstable(nn.Module):
    """RMB con pesos de ``nn.Linear`` y un sesgo de memoria acotado: base + 0,5·tanh(T·memoria)."""

    def __init__(self, entradas: int, salidas: int, decaimiento: float = 0.9) -> None:
        super().__init__()
        self.lineal = nn.Linear(entradas, salidas)
        self.transformar = nn.Linear(salidas, salidas)
        nn.init.zeros_(self.transformar.weight)
        nn.init.zeros_(self.transformar.bias)
        self.decaimiento = decaimiento
        self.register_buffer("memoria", torch.zeros(salidas))

    def forward(self, x):
        z = self.lineal(x)
        sesgo = 0.5 * torch.tanh(self.transformar(self.memoria))
        salida = F.relu(z + sesgo)
        if self.training:
            # Tensor nuevo (no en el lugar): el viejo todavía se usa para calcular el gradiente.
            media = salida.detach().mean(0).clamp(-5, 5)
            self.memoria = self.decaimiento * self.memoria + (1 - self.decaimiento) * media
        return salida


class ResonanteSuave(nn.Module):
    """sigmoide(z) · sigmoide(k · (z − fuerza)): la neurona «resuena» de a poco, y el gradiente siempre pasa.

    Arreglo: ``torch.where`` ponía 0 y gradiente 0 bajo el umbral. Además sigmoide(z) no pasa de 1, así que
    se multiplica por z cuando es positiva para no aplastar los valores grandes (como SiLU).
    """

    def __init__(self, fuerza: float = 0.5) -> None:
        super().__init__()
        self.fuerza = nn.Parameter(torch.tensor(float(fuerza)))
        self.log_k = nn.Parameter(torch.tensor(math.log(4.0)))

    def forward(self, z):
        compuerta = torch.sigmoid(torch.exp(self.log_k) * (z - self.fuerza))
        return z * torch.sigmoid(z) * compuerta + 0.01 * z


class HiperReLUSuave(nn.Module):
    """Amplifica lo que pasa el umbral, pero por 1 + 9·sigmoide(a) con a aprendida desde «sin amplificar»."""

    def __init__(self, umbral: float = 0.1) -> None:
        super().__init__()
        self.umbral = umbral
        self.a = nn.Parameter(torch.tensor(-6.0))  # 1 + 9·sigmoide(-6) ≈ 1,02

    def forward(self, z):
        factor = 1 + 9 * torch.sigmoid(self.a)
        return torch.where(z > self.umbral, z * factor, torch.where(z < 0, z * 0.01, z))


class PolinomicaRegularizada(nn.Module):
    """Neurona polinómica con términos cuadráticos de rango 4 sobre tanh(x) y penalización L2 suave."""

    def __init__(self, entradas: int, salidas: int, rango: int = 4, peso_penalizacion: float = 1e-3) -> None:
        super().__init__()
        self.lineal = nn.Linear(entradas, salidas)
        self.a = nn.Parameter(torch.randn(salidas, rango, entradas) * entradas**-0.5)
        self.b = nn.Parameter(torch.randn(salidas, rango, entradas) * entradas**-0.5 * 0.1)
        self.peso = peso_penalizacion

    def forward(self, x):
        t = torch.tanh(x)
        cuad = (torch.einsum("bi,ori->bor", t, self.a) * torch.einsum("bi,ori->bor", t, self.b)).sum(-1)
        return F.relu(self.lineal(x) + cuad)

    def perdida_extra(self, salida, objetivo):
        return self.peso * (self.a.pow(2).mean() + self.b.pow(2).mean())


def _capa(activacion_fabrica):
    from .piezas import CapaConActivacion

    return lambda e, s: CapaConActivacion(e, s, activacion_fabrica())


def piezas_reparadas():
    from .piezas import Pieza

    return [
        Pieza("atencion_dinamica", "AttentiveLinear arreglada: factores de atención que dependen de la entrada",
              "05 · reparada", AtencionDinamica),
        Pieza("rbf_entrenable", "Neuronas KNN como base radial: centros y anchos aprendidos con gradiente",
              "03 · reparada", CapaRBF),
        Pieza("hmn_sin_memoria", "HMN arreglada: mini-LSTM por neurona sin recordar la fila anterior",
              "02 · reparada", CapaHMNSinMemoria),
        Pieza("rmb_estable", "RMB arreglada: inicio de nn.Linear y memoria acotada con tanh", "02 · reparada",
              CapaRMBEstable),
        Pieza("resonante_suave", "Resonante arreglada: umbral suave y aprendido (el gradiente siempre pasa)",
              "02 · reparada", _capa(ResonanteSuave), lambda c: ResonanteSuave()),
        Pieza("hiper_relu_suave", "HiperReLU arreglada: amplificación aprendida que empieza en ×1", "02 · reparada",
              _capa(HiperReLUSuave), lambda c: HiperReLUSuave()),
        Pieza("polinomica_regularizada", "Polinómica arreglada: rango bajo, entrada acotada y penalización",
              "02 · reparada", PolinomicaRegularizada),
    ]


# ---------------------------------------------------------------- DANN (anti-gradiente bien usado)
class _InvertirGradiente(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambda_ * grad, None


class RedDANN(nn.Module):
    """Extractor → clasificador (camino normal) y extractor → GRL → «¿de qué dominio viene?» (rama aparte)."""

    def __init__(self, entradas: int, salidas: int, ocultas: int = 64) -> None:
        super().__init__()
        self.extractor = nn.Sequential(nn.Linear(entradas, ocultas), nn.ReLU(), nn.Linear(ocultas, ocultas), nn.ReLU())
        self.clasificador = nn.Linear(ocultas, salidas)
        self.dominio = nn.Sequential(nn.Linear(ocultas, ocultas), nn.ReLU(), nn.Linear(ocultas, 1))

    def forward(self, x):
        return self.clasificador(self.extractor(x))

    def dominio_de(self, x, lambda_):
        return self.dominio(_InvertirGradiente.apply(self.extractor(x), lambda_)).squeeze(1)


def datos_dominios(semilla: int = 0):
    """Dígitos 8×8. Origen: limpios, con etiqueta. Destino: dígitos movidos 1 píxel, con otro contraste y brillo
    y con ruido, **sin etiquetas** al entrenar. Entrenamiento y prueba no comparten dígitos."""
    from sklearn.datasets import load_digits

    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float32) / 16.0
    rng = np.random.default_rng(semilla)
    orden = rng.permutation(len(X))
    corte = int(0.7 * len(X))
    entreno, prueba = orden[:corte], orden[corte:]

    def mover(imgs):
        imgs = np.roll(imgs.reshape(-1, 8, 8), shift=1, axis=2)  # un píxel a la derecha
        imgs = 0.7 * imgs + 0.15  # otro contraste y otro brillo (como otra cámara)
        imgs = imgs + rng.normal(0, 0.1, imgs.shape).astype(np.float32)
        return imgs.reshape(-1, 64).astype(np.float32)

    mitad = len(entreno) // 2  # la mitad del entrenamiento es origen y la otra mitad (sin etiquetas) destino
    return {
        "origen_x": torch.tensor(X[entreno[:mitad]]), "origen_y": torch.tensor(y[entreno[:mitad]]),
        "destino_x": torch.tensor(mover(X[entreno[mitad:]])),
        "prueba_x": torch.tensor(mover(X[prueba])), "prueba_y": torch.tensor(y[prueba]),
        "prueba_origen_x": torch.tensor(X[prueba]),
    }


def entrenar_dann(datos, usar_dann: bool, semilla: int = 0, epocas: int = 30, lambda_max: float = 0.1):
    """Devuelve la exactitud en el dominio destino (y en el origen) de una red con o sin la rama DANN."""
    torch.manual_seed(semilla)
    red = RedDANN(64, 10)
    opt = torch.optim.Adam(red.parameters(), lr=1e-3)
    xo, yo, xd = datos["origen_x"], datos["origen_y"], datos["destino_x"]
    lote = 64
    pasos = max(1, len(xo) // lote)
    for epoca in range(epocas):
        # λ sube de 0 a lambda_max como en el artículo de DANN (Ganin y Lempitsky, 2015)
        p = epoca / max(1, epocas - 1)
        lambda_ = lambda_max * (2 / (1 + math.exp(-10 * p)) - 1)
        po, pd = torch.randperm(len(xo)), torch.randperm(len(xd))
        for i in range(pasos):
            io, idd = po[i * lote:(i + 1) * lote], pd[i * lote:(i + 1) * lote]
            perdida = F.cross_entropy(red(xo[io]), yo[io])
            if usar_dann and len(idd):
                dom = torch.cat([red.dominio_de(xo[io], lambda_), red.dominio_de(xd[idd], lambda_)])
                etiqueta = torch.cat([torch.zeros(len(io)), torch.ones(len(idd))])
                perdida = perdida + F.binary_cross_entropy_with_logits(dom, etiqueta)
            opt.zero_grad()
            perdida.backward()
            opt.step()
    red.eval()
    with torch.no_grad():
        destino = (red(datos["prueba_x"]).argmax(1) == datos["prueba_y"]).float().mean().item()
        origen = (red(datos["prueba_origen_x"]).argmax(1) == datos["prueba_y"]).float().mean().item()
    return destino, origen


def competir_dann(semillas: int = 5, epocas: int = 30, lambdas=(0.03, 0.1)):
    """Red normal contra la misma red con la rama DANN, varias semillas. Devuelve filas para el informe."""
    filas = {"normal": [], **{f"dann λ={l:g}": [] for l in lambdas}}
    for s in range(semillas):
        datos = datos_dominios(s)
        filas["normal"].append(entrenar_dann(datos, False, s, epocas))
        for l in lambdas:
            filas[f"dann λ={l:g}"].append(entrenar_dann(datos, True, s, epocas, lambda_max=l))
    return filas
