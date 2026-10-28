"""Cuaderno 02, segunda parte: polinómica, resonante, DMNA, GMO, CDM y CCN, más sus rivales de la industria.

Industria (referencias):
- **Dropout** (la regularización por ruido más usada): rival de DMNA, que también mete ruido al entrenar.
- **BatchNorm** (ResNet, EfficientNet) y **LayerNorm** (Transformer, Llama): rivales de AdaptiveChaosNorm (CCN).

Cada pieza dice de qué celda sale; lo que se cambió para que funcione o para que la comparación sea justa
está marcado con «Arreglo:».
"""

from __future__ import annotations

import copy
import random

import torch
import torch.nn.functional as F
from torch import nn

from .piezas import CapaConActivacion


# ---------------------------------------------------------------- Neurona polinómica dinámica (celdas 26-29)
class CapaPolinomica(nn.Module):
    """Fusión de las cuatro versiones del cuaderno en una capa de ``salidas`` neuronas polinómicas.

    Cada neurona suma:
    - términos **cuadráticos** entre pares de entradas (xᵀ·W·x) y lineales (v·x): primera versión;
    - **potencias** de cada entrada hasta el ``grado`` (Σᵢ Σₚ cₚ·xᵢᵖ): segunda y tercera versión;
    y mezcla ReLU y tanh con una proporción ``alfa`` aprendida (sigmoide de un parámetro libre).

    Arreglos:
    - Se quitó el «estado»: era un promedio de los coeficientes que no depende de la entrada (un bias más), y en
      la primera versión dependía del lote. La versión «con estado» (celda 29) usaba (Σx)ᵍʳᵃᵈᵒ, que no es el
      polinomio que describe.
    - Con pocas entradas (≤ 64) los términos cuadráticos usan la matriz completa, igual que el original. Con
      más (imágenes, texto) se usa una forma de rango 16 (como las *Factorization Machines*), porque la matriz
      completa no cabría en memoria.
    """

    def __init__(self, entradas: int, salidas: int, grado: int = 3, completa_hasta: int = 64, rango: int = 16) -> None:
        super().__init__()
        self.grado = grado
        self.completa = entradas <= completa_hasta
        if self.completa:  # W de cada neurona: matriz entradas × entradas (como ``self.w`` del original)
            self.w = nn.Parameter(torch.randn(salidas, entradas, entradas) * 0.1 / entradas)
        else:  # W ≈ Σₖ aₖ·bₖᵀ (rango bajo)
            self.a = nn.Parameter(torch.randn(salidas, rango, entradas) * entradas**-0.5)
            self.b = nn.Parameter(torch.randn(salidas, rango, entradas) * entradas**-0.5 * 0.1)
        self.lineal = nn.Linear(entradas, salidas)
        self.coef = nn.Parameter(torch.randn(salidas, grado - 1) * 0.01)  # potencias 2..grado (la 1 está en lineal)
        self.raw_alpha = nn.Parameter(torch.zeros(salidas))

    def forward(self, x):
        if self.completa:
            cuad = torch.einsum("bi,oij,bj->bo", x, self.w, x)
        else:
            cuad = (torch.einsum("bi,ori->bor", x, self.a) * torch.einsum("bi,ori->bor", x, self.b)).sum(-1)
        potencias = torch.stack([x ** p for p in range(2, self.grado + 1)], -1).sum(1)  # [b, grado-1]
        z = self.lineal(x) + cuad + potencias @ self.coef.T
        alfa = torch.sigmoid(self.raw_alpha)
        return alfa * F.relu(z) + (1 - alfa) * torch.tanh(z)


# ---------------------------------------------------------------- Neurona resonante (celdas 31-35)
class ActivacionResonante(nn.Module):
    """sigmoide(z) si z·amplificación ≥ fuerza; si no, 0 (la neurona «no resuena»).

    Arreglo: en el cuaderno la red se entrenaba con mutaciones al azar, una neurona a la vez en Python. Aquí la
    activación se usa en capas normales y se entrena con gradiente (pasa por la parte sigmoide). El método de
    mutación se compara aparte con los optimizadores (cuaderno 08).
    """

    def __init__(self, fuerza: float = 0.5, amplificacion: float = 1.0) -> None:
        super().__init__()
        self.fuerza, self.amplificacion = fuerza, amplificacion

    def forward(self, z):
        return torch.where(z * self.amplificacion >= self.fuerza, torch.sigmoid(z), torch.zeros_like(z))


# ---------------------------------------------------------------- DMNA (celda 39)
class CapaDMNA(nn.Module):
    """Lineal + ReLU que, al entrenar, suma ruido gaussiano y un empuje hacia 0 («fatiga»), con fuerza
    proporcional a √pérdida y que se apaga con el tiempo (Dynamic Model Noise Augmentation)."""

    def __init__(self, entradas: int, salidas: int, factor=0.5, decaimiento=0.999, minimo=0.01, umbral=0.1) -> None:
        super().__init__()
        self.lineal = nn.Linear(entradas, salidas)
        self.factor, self.decaimiento, self.minimo, self.umbral = factor, decaimiento, minimo, umbral
        self.perdida_actual = 0.0

    def forward(self, x):
        h = F.relu(self.lineal(x))
        if self.training and self.perdida_actual > self.umbral:
            magnitud = max(self.minimo, self.factor * self.perdida_actual**0.5)
            h = h + torch.randn_like(h) * magnitud - torch.sigmoid(h) * magnitud * 0.5
        return h

    def despues_del_paso(self, perdida, **_):
        # Arreglo: el cuaderno lo llamaba una vez por época con todo el conjunto; aquí, después de cada lote.
        self.perdida_actual = float(perdida)
        self.factor = max(self.minimo, self.factor * self.decaimiento)


class CapaDropout(nn.Module):
    """Industria: lineal + ReLU + Dropout (10 %)."""

    def __init__(self, entradas: int, salidas: int, p: float = 0.1) -> None:
        super().__init__()
        self.lineal, self.dropout = nn.Linear(entradas, salidas), nn.Dropout(p)

    def forward(self, x):
        return self.dropout(F.relu(self.lineal(x)))


# ---------------------------------------------------------------- GMO (celda 41)
class RedGMO(nn.Module):
    """La red estándar con Optimización por Mutación Global: con probabilidad 5 % por paso, perturba TODOS los
    pesos con ruido y se queda con la mutación solo si la pérdida mejora al menos 1 %.

    Arreglos:
    - El cuaderno medía con todo el conjunto (variables globales) y comparaba con la mejor pérdida vista. Con
      lotes eso no sirve: cada lote tiene otra dificultad. Aquí se mide la pérdida del **mismo lote** antes y
      después de mutar (el banco pasa ``perdida_de``, que evalúa el modelo entero en modo evaluación, sin
      dropout). Con las tareas pequeñas el lote es todo el conjunto, como en el cuaderno.
    """

    def __init__(self, entradas: int, salidas: int, ocultas: int, fuerza=0.05, probabilidad=0.05) -> None:
        super().__init__()
        self.capa1 = nn.Linear(entradas, ocultas)
        self.capa2 = nn.Linear(ocultas, ocultas)
        self.salida = nn.Linear(ocultas, salidas)
        self.fuerza, self.probabilidad = fuerza, probabilidad
        self.mutaciones = 0  # aceptadas (el banco las informa)
        self.azar = random.Random(0)

    def forward(self, x):
        return self.salida(F.relu(self.capa2(F.relu(self.capa1(x)))))

    def despues_del_paso(self, perdida_de=None, **_):
        if perdida_de is None or self.azar.random() >= self.probabilidad:
            return
        antes = float(perdida_de())
        original = copy.deepcopy(self.state_dict())
        with torch.no_grad():
            for p in self.parameters():
                p.add_(torch.randn_like(p) * self.fuerza)
        if float(perdida_de()) < antes * 0.99:
            self.mutaciones += 1
        else:
            self.load_state_dict(original)


# ---------------------------------------------------------------- CDM (celda 43)
class RedCDM(nn.Module):
    """Destilación de Conceptos: un extractor lineal «limpia» la entrada y la red estándar aprende de lo limpio.

    Arreglo importante: en el cuaderno el extractor se entrenaba con la entrada **sin ruido verdadera**, un dato
    que en la vida real no existe (y que la red estándar no recibía): la comparación no era justa. Aquí el
    extractor aprende como un autoencoder de limpieza (le llega la entrada con ruido extra y debe devolver la
    original), que es lo que la propia celda propone «en un escenario real sin X_true».
    """

    def __init__(self, entradas: int, salidas: int, ocultas: int, ruido: float = 0.1) -> None:
        super().__init__()
        self.extractor = nn.Linear(entradas, entradas)
        with torch.no_grad():  # empieza como identidad: al inicio no estorba
            self.extractor.weight.copy_(torch.eye(entradas))
            self.extractor.bias.zero_()
        self.capa1 = nn.Linear(entradas, ocultas)
        self.capa2 = nn.Linear(ocultas, ocultas)
        self.salida = nn.Linear(ocultas, salidas)
        self.ruido = ruido

    def forward(self, x):
        if self.training:
            self.entrada_actual = x.detach()
        return self.salida(F.relu(self.capa2(F.relu(self.capa1(self.extractor(x))))))

    def perdida_extra(self, salida, objetivo):
        x = self.entrada_actual
        return F.mse_loss(self.extractor(x + torch.randn_like(x) * self.ruido), x)


# ---------------------------------------------------------------- CCN (celda 47)
class HiperReLU(nn.Module):
    """HyperSensitiveReLU: multiplica por ``amplificacion`` lo que pasa el umbral; lo negativo, por 0,01.

    Igual que la ``autograd.Function`` del cuaderno (mismo gradiente), escrita con operaciones normales.
    """

    def __init__(self, amplificacion: float = 10.0, umbral: float = 0.1) -> None:
        super().__init__()
        self.amplificacion, self.umbral = amplificacion, umbral

    def forward(self, z):
        return torch.where(z > self.umbral, z * self.amplificacion, torch.where(z < 0, z * 0.01, z))


class NormaCaosAdaptativa(nn.Module):
    """AdaptiveChaosNorm: como BatchNorm, pero la varianza se infla con un «factor de turbulencia» aprendido.

    Arreglos:
    - Al evaluar, el cuaderno dividía por √varianza **sin** la turbulencia, y al entrenar con ella: la red
      veía otra escala en la prueba. Ahora se usa la misma fórmula en los dos casos.
    - El promedio de estadísticas decaía con 0,999: tras unos cientos de pasos seguía casi en (0, 1) y la
      evaluación salía mal. Se usa 0,9, como BatchNorm.
    - Con un lote de 1 ejemplo usa las estadísticas acumuladas (la varianza de 1 dato no existe).
    """

    def __init__(self, n: int, eps: float = 1e-5, decaimiento: float = 0.9) -> None:
        super().__init__()
        self.eps, self.decaimiento = eps, decaimiento
        self.gamma, self.beta = nn.Parameter(torch.ones(n)), nn.Parameter(torch.zeros(n))
        self.turbulencia = nn.Parameter(torch.ones(n) * 0.1)
        self.register_buffer("media", torch.zeros(n))
        self.register_buffer("varianza", torch.ones(n))

    def forward(self, x):
        if self.training and len(x) > 1:
            media, var = x.mean(0), x.var(0, unbiased=True)
            with torch.no_grad():
                self.media.mul_(self.decaimiento).add_(media.detach(), alpha=1 - self.decaimiento)
                self.varianza.mul_(self.decaimiento).add_(var.detach(), alpha=1 - self.decaimiento)
        else:
            media, var = self.media, self.varianza
        var = var * (1 + self.turbulencia.abs())
        return self.gamma * (x - media) / torch.sqrt(var + self.eps) + self.beta


class NormaLotes(nn.BatchNorm1d):
    """Industria: BatchNorm1d que, con un lote de 1 ejemplo al entrenar, usa las estadísticas acumuladas."""

    def forward(self, x):
        if self.training and len(x) == 1:
            return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, False, 0.0, self.eps)
        return super().forward(x)


class CapaNormalizada(nn.Module):
    """Lineal → activación → normalización (el orden de la red CCN del cuaderno, igual para las rivales)."""

    def __init__(self, entradas: int, salidas: int, activacion: nn.Module, norma: nn.Module) -> None:
        super().__init__()
        self.lineal, self.activacion, self.norma = nn.Linear(entradas, salidas), activacion, norma

    def forward(self, x):
        return self.norma(self.activacion(self.lineal(x)))


def piezas_02b():
    from .piezas import Pieza

    industria = [
        Pieza("dropout", "Lineal + ReLU + Dropout 10 % (industria)", "industria", CapaDropout, industria=True),
        Pieza("batchnorm", "Lineal + ReLU + BatchNorm (industria: ResNet)", "industria",
              lambda e, s: CapaNormalizada(e, s, nn.ReLU(), NormaLotes(s)), industria=True),
        Pieza("layernorm", "Lineal + ReLU + LayerNorm (industria: Transformer)", "industria",
              lambda e, s: CapaNormalizada(e, s, nn.ReLU(), nn.LayerNorm(s)), industria=True),
    ]
    propias = [
        Pieza("polinomica", "Neurona polinómica: términos cuadráticos y potencias, mezcla ReLU/tanh",
              "02 · neurona polinómica (4 versiones unidas)", CapaPolinomica),
        Pieza("resonante", "Solo se activa (sigmoide) si supera un umbral de «fuerza»", "02 · neurona resonante",
              lambda e, s: CapaConActivacion(e, s, ActivacionResonante()), lambda c: ActivacionResonante()),
        Pieza("dmna", "Ruido que depende de la pérdida y se apaga con el tiempo (DMNA)", "02 · DMNA", CapaDMNA),
        Pieza("gmo", "Mutación global de todos los pesos, aceptada solo si mejora (GMO)", "02 · GMO", None,
              red=lambda e, s, o: RedGMO(e, s, o)),
        Pieza("cdm", "Extractor que limpia la entrada antes de la red (CDM)", "02 · CDM", None,
              red=lambda e, s, o: RedCDM(e, s, o)),
        Pieza("hiper_relu", "ReLU que amplifica ×10 lo que pasa el umbral (de la CCN)", "02 · CCN",
              lambda e, s: CapaConActivacion(e, s, HiperReLU()), lambda c: HiperReLU()),
        Pieza("norma_caos", "Normalización con «turbulencia» aprendida, con ReLU (de la CCN)", "02 · CCN",
              lambda e, s: CapaNormalizada(e, s, nn.ReLU(), NormaCaosAdaptativa(s))),
        Pieza("ccn", "Red de Caos Controlado completa: HiperReLU + normalización con turbulencia", "02 · CCN",
              lambda e, s: CapaNormalizada(e, s, HiperReLU(), NormaCaosAdaptativa(s))),
    ]
    return industria + propias
