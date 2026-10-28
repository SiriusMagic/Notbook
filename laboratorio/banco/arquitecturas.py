"""Arquitecturas «normales» para comparar tus piezas con la tecnología de hoy.

Cada modelo es: extractor (CNN, TextCNN, Transformer, LSTM…) → cabeza = la Red
de siempre (dos capas de la pieza + salida lineal). Así una pieza compite de dos
formas a la vez:

1. **En la cabeza**: todas las piezas (IDA, HMN, RMB…) van en la parte final.
2. **Por dentro**: si la pieza es una activación (PReLU, IDA, adaptativa, seno),
   también reemplaza a ReLU dentro de la CNN. Las demás usan ReLU por dentro.

Con la pieza «estandar» cada arquitectura es exactamente la versión normal
(ReLU en todas partes): esa es la referencia a vencer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from .piezas import Pieza, Red


def _act(pieza: Pieza, canales: int) -> nn.Module:
    return pieza.activacion(canales) if pieza.activacion else nn.ReLU()


class Modelo(nn.Module):
    """Extractor de rasgos + cabeza con la pieza."""

    def __init__(self, extractor: nn.Module, rasgos: int, pieza: Pieza, salidas: int, ocultas: int) -> None:
        super().__init__()
        self.extractor = extractor
        self.cabeza = Red(pieza, rasgos, salidas, ocultas)

    def forward(self, x):
        return self.cabeza(self.extractor(x))

    def despues_del_paso(self, **contexto) -> None:
        self.cabeza.despues_del_paso(**contexto)

    def perdida_extra(self, salida, objetivo):
        """Pérdida adicional del extractor (p. ej. REINFORCE de los agentes de atención)."""
        total = self.cabeza.perdida_extra(salida, objetivo)
        if hasattr(self.extractor, "perdida_extra"):
            total = total + self.extractor.perdida_extra(salida, objetivo)
        return total


class ExtractorCNN(nn.Module):
    """CNN clásica para imágenes: [conv 3×3 → BatchNorm → activación → max-pool] × 2."""

    def __init__(self, pieza: Pieza, forma: tuple[int, int, int], canales=(32, 64)) -> None:
        super().__init__()
        self.forma = forma
        c, h, w = forma
        capas = []
        for k in canales:
            capas += [nn.Conv2d(c, k, 3, padding=1), nn.BatchNorm2d(k), _act(pieza, k), nn.MaxPool2d(2)]
            c, h, w = k, h // 2, w // 2
        self.red = nn.Sequential(*capas)
        self.rasgos = c * h * w

    def forward(self, x):
        return self.red(x.view(-1, *self.forma)).flatten(1)


class ExtractorTextoCNN(nn.Module):
    """TextCNN (Kim, 2014): embeddings → conv 1D → activación → máximo sobre el texto."""

    def __init__(self, pieza: Pieza, vocabulario: int, dim: int = 64, filtros: int = 128) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocabulario, dim, padding_idx=0)
        self.conv = nn.Conv1d(dim, filtros, 5, padding=2)
        self.act = _act(pieza, filtros)
        self.rasgos = filtros

    def forward(self, ids):
        h = self.act(self.conv(self.emb(ids).transpose(1, 2)))
        return h.amax(dim=2)


class PosicionSenoidal(nn.Module):
    """La del Transformer original (Vaswani et al., 2017); también está en la celda 70 del cuaderno 01."""

    def __init__(self, largo: int, dim: int) -> None:
        super().__init__()
        pos = torch.arange(largo).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(largo, dim)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, ids):
        return self.pe[:, : ids.size(1)]


class PosicionJerarquica(nn.Module):
    """Tu embedding posicional jerárquico (cuaderno 01, celda 69).

    - ``modo="pares"`` (el original): posición del par de sílabas + 1.ª o 2.ª del par.
    - ``modo="palabra"``: posición de la palabra + posición del token dentro de la palabra.
      Usa la marca de «empieza palabra» de cada token.
    """

    def __init__(self, largo: int, dim: int, modo: str, inicio: torch.Tensor | None) -> None:
        super().__init__()
        self.modo = modo
        self.global_ = nn.Embedding(largo + 1, dim)
        self.interna = nn.Embedding(17, dim)
        if modo == "palabra":
            self.register_buffer("inicio", inicio.bool())

    def forward(self, ids):
        n = ids.size(1)
        pos = torch.arange(n, device=ids.device).expand_as(ids)
        if self.modo == "pares":
            g, i = pos // 2, pos % 2
        else:
            empieza = self.inicio[ids] | (pos == 0)
            g = empieza.long().cumsum(1) - 1
            ultimo = torch.where(empieza, pos, torch.zeros_like(pos)).cummax(1).values
            i = pos - ultimo
        return self.global_(g.clamp(0, self.global_.num_embeddings - 1)) + self.interna(i.clamp(0, 16))


class ExtractorTransformer(nn.Module):
    """Transformer pequeño para texto: embeddings + posición → 2 capas de atención → promedio."""

    def __init__(self, pieza: Pieza, vocabulario: int, largo: int, dim: int = 64, capas: int = 2,
                 posicion: str = "aprendida", inicio: torch.Tensor | None = None) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocabulario, dim, padding_idx=0)
        self.posicion = posicion
        if posicion == "aprendida":
            self.pos = nn.Parameter(torch.zeros(1, largo, dim))
        elif posicion == "senoidal":
            self.pos_mod = PosicionSenoidal(largo, dim)
        else:
            self.pos_mod = PosicionJerarquica(largo, dim, posicion.removeprefix("jerarquica_"), inicio)
        capa = nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(capa, capas, enable_nested_tensor=False)
        self.rasgos = dim

    def forward(self, ids):
        relleno = ids == 0
        pos = self.pos[:, : ids.size(1)] if self.posicion == "aprendida" else self.pos_mod(ids)
        h = self.enc(self.emb(ids) + pos, src_key_padding_mask=relleno)
        validos = (~relleno).unsqueeze(-1).float()
        return (h * validos).sum(1) / validos.sum(1).clamp(min=1)


class ExtractorFastText(nn.Module):
    """fastText (Joulin et al., 2016): promedio de los embeddings de las palabras. Simple, rápido y difícil de batir."""

    def __init__(self, pieza: Pieza, vocabulario: int, dim: int = 64) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocabulario, dim, padding_idx=0)
        self.rasgos = dim

    def forward(self, ids):
        m = (ids != 0).unsqueeze(-1).float()
        return (self.emb(ids) * m).sum(1) / m.sum(1).clamp(min=1)


class _AgenteRL(nn.Module):
    """Base de los agentes de atención con refuerzo (cuaderno 01).

    Al entrenar, el agente *muestrea* qué palabras atender y guarda la probabilidad
    de lo que eligió; ``perdida_extra`` aplica REINFORCE: premia las elecciones que
    llevaron a acertar (recompensa 1) frente al promedio del lote (línea base).
    Al evaluar, elige lo más probable (sin azar).
    """

    log_probs: torch.Tensor | None = None
    fraccion_atendida: float = 1.0

    def perdida_extra(self, salida: torch.Tensor, objetivo: torch.Tensor) -> torch.Tensor:
        if self.log_probs is None or salida.dim() != 2:
            return salida.new_zeros(())
        recompensa = (salida.argmax(1) == objetivo).float()
        ventaja = recompensa - recompensa.mean()
        return -(ventaja * self.log_probs).mean()


class ExtractorAtencionBinaria(_AgenteRL):
    """Celda 31: un agente decide, palabra por palabra, si la lee (1) o la ignora (0); se promedian las leídas.

    Arreglos: en el cuaderno el clasificador y los embeddings nunca se entrenaban (solo el agente) y no había
    línea base en REINFORCE; aquí se entrena todo junto y se ignora el relleno.
    """

    def __init__(self, pieza: Pieza, vocabulario: int, dim: int = 64) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocabulario, dim, padding_idx=0)
        self.agente = nn.Sequential(nn.Linear(dim, 16), nn.ReLU(), nn.Linear(16, 1))
        self.rasgos = dim

    def forward(self, ids):
        e = self.emb(ids)
        validas = ids != 0
        p = torch.sigmoid(self.agente(e).squeeze(-1))
        if self.training:
            dist = torch.distributions.Bernoulli(probs=p)
            accion = dist.sample()
            self.log_probs = (dist.log_prob(accion) * validas).sum(1)
        else:
            accion = (p > 0.5).float()
            self.log_probs = None
        accion = accion * validas
        self.fraccion_atendida = float(accion.sum() / validas.sum().clamp(min=1))
        m = accion.unsqueeze(-1)
        return (e * m).sum(1) / m.sum(1).clamp(min=1)


class ExtractorAtencionNiveles(_AgenteRL):
    """Celdas 32 y 36: auto-atención + 3 cabezas; cada cabeza da a cada palabra un nivel (0 ignorar, 1 medio,
    2 fuerte); el promedio de niveles pondera los embeddings.

    Arreglos: igual que la binaria (clasificador y embeddings entrenados, línea base, sin relleno).
    """

    def __init__(self, pieza: Pieza, vocabulario: int, dim: int = 64, cabezas: int = 3) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocabulario, dim, padding_idx=0)
        self.q, self.k, self.v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.cabezas = nn.ModuleList(nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Linear(32, 3)) for _ in range(cabezas))
        self.rasgos = dim

    def forward(self, ids):
        e = self.emb(ids)
        validas = ids != 0
        q, k, v = self.q(e), self.k(e), self.v(e)
        puntajes = (q @ k.transpose(1, 2)) / math.sqrt(q.size(-1))
        puntajes = puntajes.masked_fill(~validas.unsqueeze(1), -1e9)
        ctx = torch.softmax(puntajes, -1) @ v
        niveles, logp = [], []
        for cabeza in self.cabezas:
            logits = cabeza(ctx)
            if self.training:
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                logp.append((dist.log_prob(a) * validas).sum(1))
            else:
                a = logits.argmax(-1)
            niveles.append(a.float())
        peso = torch.stack(niveles).mean(0) / 2.0 * validas
        self.log_probs = torch.stack(logp).mean(0) if logp else None
        self.fraccion_atendida = float((peso > 0).float().sum() / validas.sum().clamp(min=1))
        return (e * peso.unsqueeze(-1)).sum(1) / peso.sum(1, keepdim=True).clamp(min=1e-6)


class ExtractorAtencionHebbiana(nn.Module):
    """Celda 33: cada palabra del vocabulario tiene un peso de importancia aprendido (sigmoide); suma ponderada.

    Pese al nombre, el peso se aprende por gradiente, no por la regla de Hebb. Arreglo: embeddings aprendidos
    (el cuaderno usaba vectores al azar fijos por palabra).
    """

    def __init__(self, pieza: Pieza, vocabulario: int, dim: int = 64) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocabulario, dim, padding_idx=0)
        self.importancia = nn.Embedding(vocabulario, 1)
        nn.init.zeros_(self.importancia.weight)
        self.rasgos = dim

    def forward(self, ids):
        peso = torch.sigmoid(self.importancia(ids).squeeze(-1)) * (ids != 0)
        return (self.emb(ids) * peso.unsqueeze(-1)).sum(1) / peso.sum(1, keepdim=True).clamp(min=1e-6)


class ExtractorSerieCNN(nn.Module):
    """CNN 1D para series: dos convoluciones sobre la ventana de valores pasados."""

    def __init__(self, pieza: Pieza, largo: int, canales: int = 32) -> None:
        super().__init__()
        self.red = nn.Sequential(nn.Conv1d(1, canales, 5, padding=2), _act(pieza, canales),
                                 nn.Conv1d(canales, canales, 5, padding=2), _act(pieza, canales))
        self.rasgos = canales * largo

    def forward(self, x):
        return self.red(x.unsqueeze(1)).flatten(1)


class ExtractorLSTM(nn.Module):
    """LSTM clásica para series: lee la ventana paso a paso y se queda con su último estado."""

    def __init__(self, pieza: Pieza, oculto: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(1, oculto, batch_first=True)
        self.rasgos = oculto

    def forward(self, x):
        _, (h, _) = self.lstm(x.unsqueeze(-1))
        return h[-1]


@dataclass(frozen=True)
class Arquitectura:
    nombre: str
    descripcion: str
    datos: tuple[str, ...]  # tipos de dato que acepta: "tabla", "imagen", "texto", "serie"
    representacion: str  # "plano" (vector de números) o "ids" (palabras como números)
    crear: Callable  # (pieza, tarea) → extractor con atributo .rasgos


ARQUITECTURAS: dict[str, Arquitectura] = {a.nombre: a for a in [
    Arquitectura("densa", "red densa (la del banco desde el principio)", ("tabla", "imagen", "texto", "serie"), "plano",
                 lambda p, t: None),
    Arquitectura("cnn", "CNN clásica de 2 bloques con BatchNorm", ("imagen",), "plano",
                 lambda p, t: ExtractorCNN(p, t.forma)),
    Arquitectura("texto_cnn", "TextCNN: embeddings + convolución 1D", ("texto",), "ids",
                 lambda p, t: ExtractorTextoCNN(p, t.vocabulario)),
    Arquitectura("transformer", "Transformer pequeño (2 capas, 4 cabezas de atención)", ("texto",), "ids",
                 lambda p, t: ExtractorTransformer(p, t.vocabulario, t.forma[0])),
    Arquitectura("transformer_senoidal", "Transformer con posición senoidal (la del paper original)", ("texto",), "ids",
                 lambda p, t: ExtractorTransformer(p, t.vocabulario, t.forma[0], posicion="senoidal")),
    Arquitectura("transformer_jer_pares", "Transformer con tu posición jerárquica por pares de sílabas (cuaderno 01)",
                 ("texto",), "ids",
                 lambda p, t: ExtractorTransformer(p, t.vocabulario, t.forma[0], posicion="jerarquica_pares")),
    Arquitectura("transformer_jer_palabra", "Transformer con posición jerárquica palabra + sílaba", ("texto",), "ids",
                 lambda p, t: ExtractorTransformer(p, t.vocabulario, t.forma[0], posicion="jerarquica_palabra",
                                                   inicio=_inicio(t))),
    Arquitectura("fasttext", "fastText: promedio de embeddings (Facebook, 2016)", ("texto",), "ids",
                 lambda p, t: ExtractorFastText(p, t.vocabulario)),
    Arquitectura("atencion_binaria_rl", "tu agente que elige qué palabras leer, con refuerzo (cuaderno 01, celda 31)",
                 ("texto",), "ids", lambda p, t: ExtractorAtencionBinaria(p, t.vocabulario)),
    Arquitectura("atencion_niveles_rl", "tu agente de 3 cabezas y niveles de atención, con refuerzo (celdas 32 y 36)",
                 ("texto",), "ids", lambda p, t: ExtractorAtencionNiveles(p, t.vocabulario)),
    Arquitectura("atencion_hebbiana", "tu peso de importancia por palabra (cuaderno 01, celda 33)", ("texto",), "ids",
                 lambda p, t: ExtractorAtencionHebbiana(p, t.vocabulario)),
    Arquitectura("serie_cnn", "CNN 1D sobre la ventana de la serie", ("serie",), "plano",
                 lambda p, t: ExtractorSerieCNN(p, t.entradas)),
    Arquitectura("lstm", "LSTM clásica de 64 unidades", ("serie",), "plano",
                 lambda p, t: ExtractorLSTM(p)),
]}

# Para cada tipo de dato, contra qué tecnología normal compite cada pieza.
NATURALES = {"tabla": ("densa",), "imagen": ("cnn",), "texto": ("texto_cnn", "transformer", "fasttext"),
             "serie": ("serie_cnn", "lstm")}

# Arquitecturas que son ideas de tus cuadernos: compiten contra la mejor arquitectura normal de su tipo de dato.
INVESTIGACION = {"texto": ("atencion_binaria_rl", "atencion_niveles_rl", "atencion_hebbiana")}


def _inicio(tarea) -> torch.Tensor:
    """Qué ids empiezan palabra. Si la tarea no lo sabe, cada token es una palabra."""
    inicio = getattr(tarea, "extra", {}).get("inicio")
    return torch.as_tensor(inicio) if inicio is not None else torch.ones(tarea.vocabulario, dtype=torch.bool)


def crear_modelo(pieza: Pieza, tarea, arquitectura: str = "densa", ocultas: int = 32) -> nn.Module:
    arq = ARQUITECTURAS[arquitectura]
    if tarea.dato not in arq.datos:
        raise ValueError(f"La arquitectura «{arquitectura}» no sirve para datos de tipo «{tarea.dato}» ({tarea.nombre}).")
    if arquitectura == "densa":
        return Red(pieza, tarea.entradas, tarea.salidas, ocultas)
    extractor = arq.crear(pieza, tarea)
    return Modelo(extractor, extractor.rasgos, pieza, tarea.salidas, ocultas)


__all__ = ["ARQUITECTURAS", "INVESTIGACION", "NATURALES", "Arquitectura", "Modelo", "crear_modelo"]
