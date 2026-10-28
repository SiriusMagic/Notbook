"""Tareas con datos reales de Hugging Face (imágenes y texto).

Sirven para probar las piezas fuera de los juguetes (círculos, XOR…). Se
descargan la primera vez y quedan en caché:

- Hugging Face guarda los archivos originales en ``HF_HOME`` (por defecto
  ``~/.cache/huggingface``). En Colab conviene apuntarlo a Google Drive para no
  descargar de nuevo cada vez.
- El banco guarda además los tensores ya preparados en
  ``~/.cache/laboratorio`` (o ``LABORATORIO_CACHE``), así la segunda vez carga
  en segundos.

Para usar la misma red de las piezas (capas densas), las imágenes se aplanan
(píxeles / 255) y el texto se convierte en un vector de 2048 números con
``HashingVectorizer`` + TF-IDF. Es una forma simple y justa para todas las
piezas, no la mejor posible para cada tipo de dato.

Necesita: ``pip install datasets pillow``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .tareas import SEMILLA_DATOS, Tarea


@dataclass(frozen=True)
class FuenteHF:
    nombre: str
    repo: str  # identificador en huggingface.co/datasets
    tipo: str  # "imagen" o "texto"
    columna: str  # columna con la imagen o el texto
    descripcion: str
    clases: int
    entreno: int = 10_000  # cuántas filas tomar (al azar) de "train"
    prueba: int = 2_000  # y de "test"
    epocas: int = 50  # máximo; la parada temprana suele cortar antes
    lote: int = 128


FUENTES: dict[str, FuenteHF] = {f.nombre: f for f in [
    FuenteHF("mnist", "ylecun/mnist", "imagen", "image", "dígitos escritos a mano 28×28", 10),
    FuenteHF("fashion_mnist", "zalando-datasets/fashion_mnist", "imagen", "image", "ropa 28×28 (más difícil que MNIST)", 10),
    FuenteHF("cifar10", "uoft-cs/cifar10", "imagen", "img", "fotos a color 32×32 de 10 clases (difícil para redes densas)", 10),
    FuenteHF("resenas_es", "SetFit/amazon_reviews_multi_es", "texto", "text", "reseñas de Amazon en español, 1 a 5 estrellas", 5),
    FuenteHF("ag_news", "fancyzhx/ag_news", "texto", "text", "noticias en inglés de 4 temas", 4),
    FuenteHF("imdb", "stanfordnlp/imdb", "texto", "text", "críticas de cine en inglés, positiva o negativa", 2),
]}

DIM_TEXTO = 2048


def _carpeta_cache() -> Path:
    return Path(os.environ.get("LABORATORIO_CACHE", Path.home() / ".cache" / "laboratorio"))


def _muestra(repo: str, split: str, n: int, semilla: int):
    """Descarga el split (queda en la caché de HF), lo baraja y toma n filas.

    Se baraja ANTES de cortar: algunos conjuntos vienen ordenados por clase
    (IMDB trae primero todas las negativas) y tomar «las primeras n» daría una sola clase.
    """
    from datasets import load_dataset

    ds = load_dataset(repo, split=split)
    ds = ds.shuffle(seed=semilla)
    return ds.select(range(min(n, len(ds))))


def imagenes_a_matriz(imagenes) -> np.ndarray:
    filas = [np.asarray(img, dtype=np.float32).reshape(-1) / 255.0 for img in imagenes]
    return np.stack(filas)


def textos_a_matriz(textos_entreno: list[str], textos_prueba: list[str], dim: int = DIM_TEXTO):
    """Texto → vector fijo. El TF-IDF se ajusta solo con entrenamiento (sin mirar la prueba)."""
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

    hv = HashingVectorizer(n_features=dim, alternate_sign=False, ngram_range=(1, 2), norm=None)
    tfidf = TfidfTransformer()
    xe = tfidf.fit_transform(hv.transform(textos_entreno))
    xp = tfidf.transform(hv.transform(textos_prueba))
    return xe.toarray().astype(np.float32), xp.toarray().astype(np.float32)


def armar_tarea(fuente: FuenteHF, filas_entreno, filas_prueba) -> Tarea:
    """Convierte filas de Hugging Face (dicts con la columna y «label») en una Tarea del banco."""
    col = fuente.columna
    ye = np.asarray([f["label"] for f in filas_entreno], dtype=np.int64)
    yp = np.asarray([f["label"] for f in filas_prueba], dtype=np.int64)
    if fuente.tipo == "imagen":
        xe = imagenes_a_matriz(f[col] for f in filas_entreno)
        xp = imagenes_a_matriz(f[col] for f in filas_prueba)
        media, std = xe.mean(0), xe.std(0) + 1e-6
        xe, xp = (xe - media) / std, (xp - media) / std
    else:
        xe, xp = textos_a_matriz([f[col] for f in filas_entreno], [f[col] for f in filas_prueba])
    return Tarea(fuente.nombre, "clasificacion", torch.from_numpy(xe), torch.from_numpy(ye),
                 torch.from_numpy(xp), torch.from_numpy(yp), xe.shape[1], fuente.clases, fuente.epocas,
                 f"{fuente.descripcion} ({len(ye):,} para entrenar, {len(yp):,} de prueba)", lote=fuente.lote)


def cargar_real(nombre: str, entreno: int | None = None, prueba: int | None = None) -> Tarea:
    """Descarga (solo la primera vez), prepara y guarda en caché una tarea real."""
    fuente = FUENTES[nombre]
    ne, np_ = entreno or fuente.entreno, prueba or fuente.prueba
    ruta = _carpeta_cache() / f"{nombre}_{ne}_{np_}_{SEMILLA_DATOS}.pt"
    if ruta.exists():
        d = torch.load(ruta, weights_only=True)  # solo tensores, nunca código
        return Tarea(nombre, "clasificacion", d["xe"], d["ye"], d["xp"], d["yp"], d["xe"].shape[1],
                     fuente.clases, fuente.epocas, d["descripcion"], lote=fuente.lote)
    tarea = armar_tarea(fuente, _muestra(fuente.repo, "train", ne, SEMILLA_DATOS),
                        _muestra(fuente.repo, "test", np_, SEMILLA_DATOS))
    ruta.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"xe": tarea.x_entreno, "ye": tarea.y_entreno, "xp": tarea.x_prueba, "yp": tarea.y_prueba,
                "descripcion": tarea.descripcion}, ruta)
    return tarea
