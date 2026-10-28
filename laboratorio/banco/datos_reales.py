"""Tareas con datos reales de Hugging Face: imágenes, texto y series de tiempo.

Se descargan la primera vez y quedan en caché:

- Hugging Face guarda los archivos originales en ``HF_HOME`` (por defecto
  ``~/.cache/huggingface``; en Colab el cuaderno 10 usa ``/content/hf``).
- El banco guarda los tensores ya preparados en ``LABORATORIO_CACHE`` (por
  defecto ``~/.cache/laboratorio``; en Colab ``/content/cache``), así la
  segunda vez carga en segundos.

Cómo se preparan:

- **Imágenes**: píxeles normalizados. La red densa las ve aplanadas; la CNN, con su forma.
- **Texto**: dos versiones.
  - ``plano``: vector TF-IDF de 2048 números, para la red densa.
  - ``ids``: las palabras como números, para TextCNN y Transformer. Usa las
    20 000 más frecuentes del entrenamiento y los primeros 256 tokens.
- **Series**: ventanas de valores pasados → predecir el siguiente. Se dividen
  en orden de tiempo y se normalizan solo con la parte de entrenamiento.

Necesita: ``pip install datasets pillow``.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .tareas import SEMILLA_DATOS, Tarea

DIM_TEXTO = 2048
VOCABULARIO = 20_000
LARGO_TEXTO = 256


@dataclass(frozen=True)
class FuenteHF:
    nombre: str
    repo: str  # identificador en huggingface.co/datasets
    tipo: str  # "imagen", "texto" o "serie"
    columna: str  # imagen / texto; en series, la serie (id) a usar
    descripcion: str
    clases: int = 0
    entreno: int = 10_000  # cuántas filas (o ventanas) tomar para entrenar
    prueba: int = 2_000
    epocas: int = 50  # máximo; la parada temprana suele cortar antes
    lote: int = 128
    archivo: str = ""  # series: archivo parquet dentro del repo
    ventana: int = 48  # series: cuántos valores pasados ve el modelo


_CHRONOS = "autogluon/chronos_datasets"

FUENTES: dict[str, FuenteHF] = {f.nombre: f for f in [
    FuenteHF("mnist", "ylecun/mnist", "imagen", "image", "dígitos escritos a mano 28×28", 10),
    FuenteHF("fashion_mnist", "zalando-datasets/fashion_mnist", "imagen", "image", "ropa 28×28 (más difícil que MNIST)", 10),
    FuenteHF("cifar10", "uoft-cs/cifar10", "imagen", "img", "fotos a color 32×32 de 10 clases", 10),
    FuenteHF("resenas_es", "SetFit/amazon_reviews_multi_es", "texto", "text", "reseñas de Amazon en español, 1 a 5 estrellas", 5),
    FuenteHF("ag_news", "fancyzhx/ag_news", "texto", "text", "noticias en inglés de 4 temas", 4),
    FuenteHF("imdb", "stanfordnlp/imdb", "texto", "text", "críticas de cine en inglés, positiva o negativa", 2),
    FuenteHF("electricidad", _CHRONOS, "serie", "COAST", "demanda eléctrica por hora en Texas (ERCOT, región costa)",
             archivo="ercot/train-00000-of-00001.parquet", ventana=48),
    FuenteHF("tipo_cambio", _CHRONOS, "serie", "currency_1", "tipo de cambio diario de una moneda (casi un paseo al azar: muy difícil)",
             archivo="exchange_rate/train-00000-of-00001.parquet", ventana=30, entreno=5_000, prueba=1_500),
    FuenteHF("rio", _CHRONOS, "serie", "T000000", "caudal diario del río Saugeen (Canadá)",
             archivo="monash_saugeenday/train-00000-of-00001.parquet", ventana=30),
]}


def _carpeta_cache() -> Path:
    return Path(os.environ.get("LABORATORIO_CACHE", Path.home() / ".cache" / "laboratorio"))


# ---------------------------------------------------------------- Descarga
def _muestra(repo: str, split: str, n: int, semilla: int):
    """Descarga el split (queda en la caché de HF), lo baraja y toma n filas.

    Se baraja ANTES de cortar: algunos conjuntos vienen ordenados por clase
    (IMDB trae primero todas las negativas) y tomar «las primeras n» daría una sola clase.
    """
    from datasets import load_dataset

    ds = load_dataset(repo, split=split)
    ds = ds.shuffle(seed=semilla)
    return ds.select(range(min(n, len(ds))))


def _serie(repo: str, archivo: str, serie_id: str) -> np.ndarray:
    """Descarga el parquet (una vez) y devuelve la serie pedida, con los huecos rellenados."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    tabla = pq.read_table(hf_hub_download(repo, archivo, repo_type="dataset")).to_pydict()
    columna = next(c for c in tabla if c not in ("id", "timestamp"))
    y = np.asarray(tabla[columna][tabla["id"].index(serie_id)], dtype=np.float64)
    huecos = np.isnan(y)
    if huecos.any():  # rellena cada hueco con el último valor conocido
        idx = np.where(~huecos, np.arange(len(y)), 0)
        np.maximum.accumulate(idx, out=idx)
        y = y[idx]
    return y


def textos_reales(nombre: str, entreno: int | None = None, prueba: int | None = None):
    """Textos crudos y etiquetas de una tarea de texto: (textos_e, y_e, textos_p, y_p)."""
    fuente = FUENTES[nombre]
    if fuente.tipo != "texto":
        raise ValueError(f"{nombre} no es una tarea de texto")
    ne = entreno or int(os.environ.get("LABORATORIO_ENTRENO", 0)) or fuente.entreno
    np_ = prueba or int(os.environ.get("LABORATORIO_PRUEBA", 0)) or fuente.prueba
    fe = _muestra(fuente.repo, "train", ne, SEMILLA_DATOS)
    fp = _muestra(fuente.repo, "test", np_, SEMILLA_DATOS)
    col = fuente.columna
    return ([f[col] for f in fe], [f["label"] for f in fe], [f[col] for f in fp], [f["label"] for f in fp])


# ---------------------------------------------------------------- Imágenes y texto
def imagenes_a_matriz(imagenes) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Devuelve (filas aplanadas, forma (canales, alto, ancho))."""
    arr = np.stack([np.asarray(img, dtype=np.float32) / 255.0 for img in imagenes])
    if arr.ndim == 3:  # escala de grises: (N, alto, ancho)
        forma = (1, arr.shape[1], arr.shape[2])
    else:  # color: (N, alto, ancho, canales) → canales primero
        arr = arr.transpose(0, 3, 1, 2)
        forma = tuple(arr.shape[1:])
    return arr.reshape(len(arr), -1), forma


def textos_a_matriz(textos_entreno: list[str], textos_prueba: list[str], dim: int = DIM_TEXTO):
    """Texto → vector TF-IDF fijo. Se ajusta solo con entrenamiento (sin mirar la prueba)."""
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

    hv = HashingVectorizer(n_features=dim, alternate_sign=False, ngram_range=(1, 2), norm=None)
    tfidf = TfidfTransformer()
    xe = tfidf.fit_transform(hv.transform(textos_entreno))
    xp = tfidf.transform(hv.transform(textos_prueba))
    return xe.toarray().astype(np.float32), xp.toarray().astype(np.float32)


_PALABRA = re.compile(r"\w+|[^\w\s]")


def textos_a_ids(textos_entreno: list[str], textos_prueba: list[str], vocabulario: int = VOCABULARIO,
                 largo: int = LARGO_TEXTO):
    """Texto → palabras numeradas. 0 = relleno, 1 = palabra desconocida. Vocabulario solo del entrenamiento."""
    trozos_e = [_PALABRA.findall(t.lower()) for t in textos_entreno]
    cuentas = Counter(w for ws in trozos_e for w in ws)
    indice = {w: i + 2 for i, (w, _) in enumerate(cuentas.most_common(vocabulario - 2))}

    def codificar(trozos):
        m = np.zeros((len(trozos), largo), dtype=np.int64)
        for i, ws in enumerate(trozos):
            ids = [indice.get(w, 1) for w in ws[:largo]]
            m[i, : len(ids)] = ids
        return m

    return codificar(trozos_e), codificar([_PALABRA.findall(t.lower()) for t in textos_prueba]), len(indice) + 2


def armar_tarea(fuente: FuenteHF, filas_entreno, filas_prueba, representacion: str = "plano") -> Tarea:
    """Convierte filas de Hugging Face (dicts con la columna y «label») en una Tarea del banco."""
    col = fuente.columna
    ye = np.asarray([f["label"] for f in filas_entreno], dtype=np.int64)
    yp = np.asarray([f["label"] for f in filas_prueba], dtype=np.int64)
    forma = vocab = None
    if fuente.tipo == "imagen":
        xe, forma = imagenes_a_matriz(f[col] for f in filas_entreno)
        xp, _ = imagenes_a_matriz(f[col] for f in filas_prueba)
        media, std = xe.mean(0), xe.std(0) + 1e-6
        xe, xp = (xe - media) / std, (xp - media) / std
        x_e, x_p = torch.from_numpy(xe), torch.from_numpy(xp)
    elif representacion == "ids":
        xe, xp, vocab = textos_a_ids([f[col] for f in filas_entreno], [f[col] for f in filas_prueba])
        forma = (xe.shape[1],)
        x_e, x_p = torch.from_numpy(xe), torch.from_numpy(xp)
    else:
        xe, xp = textos_a_matriz([f[col] for f in filas_entreno], [f[col] for f in filas_prueba])
        x_e, x_p = torch.from_numpy(xe), torch.from_numpy(xp)
    return Tarea(fuente.nombre, "clasificacion", x_e, torch.from_numpy(ye), x_p, torch.from_numpy(yp), xe.shape[1],
                 fuente.clases, fuente.epocas, f"{fuente.descripcion} ({len(ye):,} para entrenar, {len(yp):,} de prueba)",
                 lote=fuente.lote, dato=fuente.tipo, forma=forma, vocabulario=vocab, representacion=representacion)


# ---------------------------------------------------------------- Series
def armar_serie(fuente: FuenteHF, valores: np.ndarray) -> Tarea:
    """Serie → ventanas (valores pasados → siguiente valor), en orden de tiempo y sin fuga de futuro.

    Se usan los últimos ``entreno + prueba + ventana`` puntos. La normalización usa
    solo la parte de entrenamiento. También se calcula el error de la predicción más
    simple (repetir el último valor): un modelo que no la supera no sirve.
    """
    v = fuente.ventana
    n_e, n_p = fuente.entreno, fuente.prueba
    s = np.asarray(valores, dtype=np.float64)[-(n_e + n_p + v):]
    n_e = min(n_e, len(s) - v - max(1, int((len(s) - v) * 0.2)))  # series cortas: deja al menos 20 % para prueba
    media, std = s[: n_e + v].mean(), s[: n_e + v].std() + 1e-8
    s = ((s - media) / std).astype(np.float32)
    x = np.lib.stride_tricks.sliding_window_view(s[:-1], v).copy()
    y = s[v:].reshape(-1, 1)
    xe, ye, xp, yp = x[:n_e], y[:n_e], x[n_e:], y[n_e:]
    ingenuo = float(np.mean((xp[:, -1:] - yp) ** 2))
    return Tarea(fuente.nombre, "regresion", torch.from_numpy(xe), torch.from_numpy(ye), torch.from_numpy(xp),
                 torch.from_numpy(yp), v, 1, fuente.epocas,
                 f"{fuente.descripcion} ({len(ye):,} ventanas de {v} para entrenar, {len(yp):,} de prueba)",
                 lote=fuente.lote, dato="serie", ingenuo=ingenuo)


# ---------------------------------------------------------------- Carga con caché
_CAMPOS = ("entradas", "salidas", "epocas", "descripcion", "lote", "dato", "forma", "vocabulario", "representacion", "ingenuo")


def cargar_real(nombre: str, entreno: int | None = None, prueba: int | None = None, representacion: str = "plano") -> Tarea:
    """Descarga (solo la primera vez), prepara y guarda en caché una tarea real."""
    fuente = FUENTES[nombre]
    # La línea de comandos (--entreno / --prueba / --rapido) los pasa por variables de entorno.
    entreno = entreno or int(os.environ.get("LABORATORIO_ENTRENO", 0)) or None
    prueba = prueba or int(os.environ.get("LABORATORIO_PRUEBA", 0)) or None
    if entreno or prueba:
        from dataclasses import replace

        fuente = replace(fuente, entreno=entreno or fuente.entreno, prueba=prueba or fuente.prueba)
    if fuente.tipo != "texto":
        representacion = "plano"
    ruta = _carpeta_cache() / f"{nombre}_{fuente.entreno}_{fuente.prueba}_{representacion}_{SEMILLA_DATOS}_v2.pt"
    if ruta.exists():
        d = torch.load(ruta, weights_only=True)  # solo tensores y números, nunca código
        extra = {k: d[k] for k in _CAMPOS}
        extra["forma"] = tuple(extra["forma"]) if extra["forma"] is not None else None
        return Tarea(nombre, d["tipo"], d["xe"], d["ye"], d["xp"], d["yp"], **extra)
    if fuente.tipo == "serie":
        tarea = armar_serie(fuente, _serie(fuente.repo, fuente.archivo, fuente.columna))
    else:
        tarea = armar_tarea(fuente, _muestra(fuente.repo, "train", fuente.entreno, SEMILLA_DATOS),
                            _muestra(fuente.repo, "test", fuente.prueba, SEMILLA_DATOS), representacion)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"tipo": tarea.tipo, "xe": tarea.x_entreno, "ye": tarea.y_entreno, "xp": tarea.x_prueba,
                "yp": tarea.y_prueba, **{k: getattr(tarea, k) for k in _CAMPOS}}, ruta)
    return tarea
