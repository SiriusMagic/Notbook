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


class ExtractorTransformer(nn.Module):
    """Transformer pequeño para texto: embeddings + posición → 2 capas de atención → promedio."""

    def __init__(self, pieza: Pieza, vocabulario: int, largo: int, dim: int = 64, capas: int = 2) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocabulario, dim, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, largo, dim))
        capa = nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(capa, capas, enable_nested_tensor=False)
        self.rasgos = dim

    def forward(self, ids):
        relleno = ids == 0
        h = self.enc(self.emb(ids) + self.pos[:, : ids.size(1)], src_key_padding_mask=relleno)
        validos = (~relleno).unsqueeze(-1).float()
        return (h * validos).sum(1) / validos.sum(1).clamp(min=1)


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
    Arquitectura("serie_cnn", "CNN 1D sobre la ventana de la serie", ("serie",), "plano",
                 lambda p, t: ExtractorSerieCNN(p, t.entradas)),
    Arquitectura("lstm", "LSTM clásica de 64 unidades", ("serie",), "plano",
                 lambda p, t: ExtractorLSTM(p)),
]}

# Para cada tipo de dato, contra qué tecnología normal compite cada pieza.
NATURALES = {"tabla": ("densa",), "imagen": ("cnn",), "texto": ("texto_cnn", "transformer"), "serie": ("serie_cnn", "lstm")}


def crear_modelo(pieza: Pieza, tarea, arquitectura: str = "densa", ocultas: int = 32) -> nn.Module:
    arq = ARQUITECTURAS[arquitectura]
    if tarea.dato not in arq.datos:
        raise ValueError(f"La arquitectura «{arquitectura}» no sirve para datos de tipo «{tarea.dato}» ({tarea.nombre}).")
    if arquitectura == "densa":
        return Red(pieza, tarea.entradas, tarea.salidas, ocultas)
    extractor = arq.crear(pieza, tarea)
    return Modelo(extractor, extractor.rasgos, pieza, tarea.salidas, ocultas)


__all__ = ["ARQUITECTURAS", "NATURALES", "Arquitectura", "Modelo", "crear_modelo"]
