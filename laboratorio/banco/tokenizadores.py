"""Competencia de tokenizadores (cuaderno 01): tus sílabas contra los de la industria.

Tokenizadores:

| Nombre | Qué es | Quién lo usa |
|---|---|---|
| ``silabas`` | sílabas por reglas del español (``laboratorio/silabas.py``) | tu idea, versión corregida |
| ``silabas_cuaderno`` | la heurística original del cuaderno 01 (``SimpleSyllableTokenizer``) | tu idea, tal cual |
| ``palabras`` | una palabra = un token | modelos clásicos (word2vec, fastText) |
| ``caracteres`` | una letra = un token | modelos por caracteres (ByT5, CANINE) |
| ``bpe`` | Byte-Pair Encoding entrenado con los mismos textos | familia GPT, Llama, Mistral |
| ``unigram`` | SentencePiece Unigram entrenado con los mismos textos | T5, Gemma, XLNet |
| ``gpt4o`` | ``tiktoken`` ``o200k_base``, ya entrenado por OpenAI | GPT-4o |
| ``rl_cuaderno`` / ``rl_arreglado`` | agente con Q-learning que decide dónde cortar (``tokenizador_rl.py``) | tu idea (celda 12) |

Todos pasan el texto a minúsculas y usan **el mismo tamaño de vocabulario**
(las 8 000 unidades más frecuentes del entrenamiento; el resto cuenta como
desconocida). Así lo único que cambia es **cómo se corta el texto**.

Los tokens que empiezan palabra se marcan; eso permite el embedding posicional
jerárquico (posición de la palabra + posición de la sílaba dentro de la palabra).
"""

from __future__ import annotations

import re
import time
import unicodedata
from collections import Counter

import numpy as np

from ..silabas import silabas

INICIO = "▁"  # marca de «empieza palabra», como en SentencePiece
_TROZOS = re.compile(r"\w+|[^\w\s]")
_PALABRA = re.compile(r"[a-záéíóúüñ]+")


# ---------------------------------------------------------------- Heurística original del cuaderno 01 (celda 69)
def _silabear_cuaderno(palabra: str) -> list[str]:
    """``SimpleSyllableTokenizer._syllabify_word_core`` + la regla «≤ 3 letras no se separa», sin cambios."""
    if len(unicodedata.normalize("NFKD", palabra).encode("ascii", "ignore").decode()) <= 3:
        return [palabra]
    vocales = "aeiouáéíóúü"
    es_v = lambda c: c in vocales  # noqa: E731
    es_c = lambda c: c not in vocales and c.isalpha()  # noqa: E731

    def grupo_valido(c1, c2):
        return (c1 == "p" and c2 == "s") or (c1 in "bcdfgpt" and c2 in "lr")

    i, out, w = 0, [], palabra
    while i < len(w):
        ini = i
        while i < len(w) and es_c(w[i]):
            i += 1
        while i < len(w) and es_v(w[i]):
            i += 1
        if i == ini:
            out.append(w[ini:])
            break
        fin = i
        if i < len(w) and es_c(w[i]):
            if i + 1 < len(w) and es_v(w[i + 1]):
                if w[i:i + 2] in {"ch", "ll", "rr"}:
                    fin = i + 2
            elif i + 1 < len(w) and es_c(w[i + 1]):
                fin = i + 2 if grupo_valido(w[i], w[i + 1]) else i + 1
            else:
                fin = i + 1
        if w[ini:fin]:
            out.append(w[ini:fin])
        i = fin
    return [s for s in out if s] or [palabra]


# ---------------------------------------------------------------- Tokenizadores
class Tokenizador:
    nombre = "base"
    descripcion = ""

    def entrenar(self, textos: list[str]) -> None:  # noqa: B027 - opcional
        pass

    def tokens(self, texto: str) -> list[str]:
        raise NotImplementedError


class PorTrozos(Tokenizador):
    """Divide en palabras y signos; cada trozo se corta con ``cortar`` (la 1.ª parte lleva la marca de inicio)."""

    def __init__(self, nombre: str, descripcion: str, cortar) -> None:
        self.nombre, self.descripcion, self.cortar = nombre, descripcion, cortar

    def tokens(self, texto: str) -> list[str]:
        out = []
        for trozo in _TROZOS.findall(texto.lower()):
            partes = self.cortar(trozo) if _PALABRA.fullmatch(trozo) else [trozo]
            out.append(INICIO + partes[0])
            out.extend(partes[1:])
        return out


class Caracteres(Tokenizador):
    nombre, descripcion = "caracteres", "una letra = un token (ByT5, CANINE)"

    def tokens(self, texto: str) -> list[str]:
        out = []
        for trozo in _TROZOS.findall(texto.lower()):
            out.append(INICIO + trozo[0])
            out.extend(trozo[1:])
        return out


class BPE(Tokenizador):
    nombre, descripcion = "bpe", "Byte-Pair Encoding entrenado con los mismos textos (GPT, Llama, Mistral)"

    def __init__(self, vocabulario: int = 8000) -> None:
        self.vocabulario = vocabulario

    def entrenar(self, textos: list[str]) -> None:
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers

        self.tok = Tokenizer(models.BPE(unk_token="[UNK]"))
        self.tok.pre_tokenizer = pre_tokenizers.Metaspace()
        trainer = trainers.BpeTrainer(vocab_size=self.vocabulario, special_tokens=["[PAD]", "[UNK]"], show_progress=False)
        self.tok.train_from_iterator((t.lower() for t in textos), trainer=trainer)

    def tokens(self, texto: str) -> list[str]:
        return self.tok.encode(texto.lower()).tokens


class Unigram(Tokenizador):
    nombre, descripcion = "unigram", "SentencePiece Unigram entrenado con los mismos textos (T5, Gemma)"

    def __init__(self, vocabulario: int = 8000) -> None:
        self.vocabulario = vocabulario

    def entrenar(self, textos: list[str]) -> None:
        import io

        import sentencepiece as spm

        modelo = io.BytesIO()
        spm.SentencePieceTrainer.train(sentence_iterator=iter([t.lower() for t in textos]), model_writer=modelo,
                                       vocab_size=self.vocabulario, model_type="unigram", character_coverage=1.0,
                                       hard_vocab_limit=False, minloglevel=3)
        self.sp = spm.SentencePieceProcessor(model_proto=modelo.getvalue())

    def tokens(self, texto: str) -> list[str]:
        return self.sp.encode(texto.lower(), out_type=str)


class SilabasBPE(Tokenizador):
    """Mejora 1 de tus sílabas: **BPE sobre sílabas**. Parte de las sílabas (no de las letras) y junta las que
    aparecen seguidas a menudo («ca»+«li»+«dad» → «calidad»), sin cruzar nunca el borde entre palabras.

    Así los tokens siempre caen en un borde de sílaba (respetan la pronunciación) y las palabras frecuentes
    quedan enteras (secuencias cortas, como BPE). Cada sílaba se escribe como un carácter privado de Unicode
    para reusar el BPE de ``tokenizers``. Una sílaba nunca vista se escribe con sus letras.
    """

    nombre, descripcion = "silabas_bpe", "BPE sobre sílabas: junta sílabas frecuentes sin cortar ninguna (mejora tuya)"
    _BASE = 0xF0000  # zona de uso privado de Unicode

    def __init__(self, vocabulario: int = 8000) -> None:
        self.vocabulario = vocabulario

    def _codificar(self, texto: str, crear: bool) -> str:
        palabras = []
        for trozo in _TROZOS.findall(texto.lower()):
            partes = silabas(trozo) if _PALABRA.fullmatch(trozo) else [trozo]
            letras = []
            for sil in partes:
                if sil not in self.a_letra:
                    if not crear:  # sílaba nunca vista: se escribe con sus letras (como el respaldo de BPE)
                        letras.append(sil)
                        continue
                    self.a_letra[sil] = chr(self._BASE + len(self.a_letra))
                    self.de_letra[self.a_letra[sil]] = sil
                letras.append(self.a_letra[sil])
            palabras.append("".join(letras))
        return " ".join(palabras)

    def entrenar(self, textos: list[str]) -> None:
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers

        self.a_letra, self.de_letra = {}, {}
        codificados = [self._codificar(t, True) for t in textos]
        self.tok = Tokenizer(models.BPE(unk_token="[UNK]"))
        self.tok.pre_tokenizer = pre_tokenizers.Metaspace()
        trainer = trainers.BpeTrainer(vocab_size=max(self.vocabulario, len(self.a_letra) + 300),
                                      special_tokens=["[PAD]", "[UNK]"], show_progress=False,
                                      initial_alphabet=list("abcdefghijklmnñopqrstuvwxyzáéíóúü0123456789"))
        self.tok.train_from_iterator(codificados, trainer=trainer)

    def tokens(self, texto: str) -> list[str]:
        out = []
        for t in self.tok.encode(self._codificar(texto, False)).tokens:
            out.append("".join(self.de_letra.get(c, c) for c in t))
        return out


class SilabasYPalabras(Tokenizador):
    """Mejora 2 de tus sílabas: las ``frecuentes`` palabras más comunes del entrenamiento van enteras y el resto
    se corta en sílabas. Las palabras comunes («bueno», «calidad») no se rompen; las raras se arman con sílabas
    conocidas en vez de quedar como desconocidas."""

    nombre, descripcion = "silabas_palabras", "palabras frecuentes enteras y el resto en sílabas (mejora tuya)"

    def __init__(self, frecuentes: int = 3000) -> None:
        self.frecuentes = frecuentes

    def entrenar(self, textos: list[str]) -> None:
        cuentas = Counter(w for t in textos for w in _PALABRA.findall(t.lower()))
        self.enteras = {w for w, _ in cuentas.most_common(self.frecuentes)}

    def tokens(self, texto: str) -> list[str]:
        out = []
        for trozo in _TROZOS.findall(texto.lower()):
            if _PALABRA.fullmatch(trozo) and trozo not in self.enteras:
                partes = silabas(trozo)
            else:
                partes = [trozo]
            out.append(INICIO + partes[0])
            out.extend(partes[1:])
        return out


class GPT4o(Tokenizador):
    nombre, descripcion = "gpt4o", "tiktoken o200k_base, el de GPT-4o (ya entrenado por OpenAI)"

    def entrenar(self, textos: list[str]) -> None:
        import tiktoken

        self.enc = tiktoken.get_encoding("o200k_base")

    def tokens(self, texto: str) -> list[str]:
        out = []
        for i in self.enc.encode(texto.lower()):
            b = self.enc.decode_single_token_bytes(i)
            out.append((INICIO if b.startswith(b" ") else "") + str(i))
        return out


def tokenizadores_disponibles() -> dict[str, Tokenizador]:
    from .tokenizador_rl import TokenizadorRL

    return {t.nombre: t for t in [
        PorTrozos("silabas", "sílabas por reglas del español (tu idea, corregida)", silabas),
        PorTrozos("silabas_cuaderno", "heurística original del cuaderno 01 (tu idea, tal cual)", _silabear_cuaderno),
        PorTrozos("palabras", "una palabra = un token (word2vec, fastText)", lambda w: [w]),
        SilabasBPE(), SilabasYPalabras(),
        Caracteres(), BPE(), Unigram(), GPT4o(), TokenizadorRL(arreglado=False), TokenizadorRL(arreglado=True),
    ]}


# ---------------------------------------------------------------- Vocabulario común y métricas
class Vocabulario:
    """Las ``tamano`` unidades más frecuentes del entrenamiento. 0 = relleno, 1 = desconocida."""

    def __init__(self, listas: list[list[str]], tamano: int = 8000) -> None:
        cuentas = Counter(t for ts in listas for t in ts)
        self.indice = {t: i + 2 for i, (t, _) in enumerate(cuentas.most_common(tamano - 2))}
        self.tamano = len(self.indice) + 2
        self.inicio = np.zeros(self.tamano, dtype=bool)  # ¿el token empieza palabra?
        for t, i in self.indice.items():
            self.inicio[i] = t.startswith(INICIO)
        self.inicio[1] = True  # una desconocida cuenta como inicio (no se sabe)

    def codificar(self, listas: list[list[str]], largo: int) -> np.ndarray:
        m = np.zeros((len(listas), largo), dtype=np.int64)
        for f, ts in enumerate(listas):
            ids = [self.indice.get(t, 1) for t in ts[:largo]]
            m[f, : len(ids)] = ids
        return m


def metricas_intrinsecas(tok: Tokenizador, vocab: Vocabulario, textos: list[str], largo: int) -> dict:
    """Fertilidad, desconocidas, truncados y velocidad sobre textos que el tokenizador no vio al entrenar."""
    t0 = time.perf_counter()
    listas = [tok.tokens(t) for t in textos]
    segundos = time.perf_counter() - t0
    palabras = sum(len(re.findall(r"\w+", t)) for t in textos)
    total = sum(len(ts) for ts in listas)
    desconocidas = sum(1 for ts in listas for t in ts if t not in vocab.indice)
    return {
        "fertilidad": round(total / max(palabras, 1), 3),  # tokens por palabra (menos = secuencias más cortas)
        "tokens_por_texto": round(total / len(textos), 1),
        "desconocidas_pct": round(100 * desconocidas / max(total, 1), 2),
        "truncados_pct": round(100 * sum(len(ts) > largo for ts in listas) / len(textos), 1),
        "palabras_por_s": round(palabras / segundos),
        "vocabulario": vocab.tamano,
    }


# ---------------------------------------------------------------- Competencia
def tarea_tokenizada(nombre: str, tok: Tokenizador, textos_e, y_e, textos_p, y_p, clases: int,
                     largo: int = 256, tamano: int = 8000):
    """Tarea del banco con el texto cortado por ``tok``; devuelve (tarea, métricas intrínsecas)."""
    import torch

    from .tareas import Tarea

    tok.entrenar(textos_e)
    listas_e = [tok.tokens(t) for t in textos_e]
    vocab = Vocabulario(listas_e, tamano)
    intrinsecas = metricas_intrinsecas(tok, vocab, textos_p, largo)
    xe = torch.from_numpy(vocab.codificar(listas_e, largo))
    xp = torch.from_numpy(vocab.codificar([tok.tokens(t) for t in textos_p], largo))
    tarea = Tarea(f"{nombre}@{tok.nombre}", "clasificacion", xe, torch.tensor(y_e), xp, torch.tensor(y_p), largo,
                  clases, 50, f"{nombre} cortado con {tok.nombre}", lote=128, dato="texto", forma=(largo,),
                  vocabulario=vocab.tamano, representacion="ids", extra={"inicio": torch.from_numpy(vocab.inicio)})
    return tarea, intrinsecas


def competir_tokenizadores(nombre: str = "resenas_es", tokenizadores: list[str] | None = None,
                           arquitecturas=("texto_cnn", "transformer"), cambios: dict | None = None,
                           posiciones: bool = True, aviso=print, textos=None) -> dict:
    """Corre la competencia. Devuelve {"intrinsecas": {...}, "filas": [...]} para ``informe_tokenizadores``.

    ``posiciones=True`` añade, con las sílabas, el Transformer con posición senoidal y tus dos jerárquicas.
    ``textos`` permite pasar (textos_e, y_e, textos_p, y_p) ya cargados (sirve para las pruebas, sin internet).
    """
    from dataclasses import replace

    from .datos_reales import FUENTES, textos_reales
    from .evaluar import CONFIG_REAL, ajustar_multiples, comparar, correr

    textos_e, y_e, textos_p, y_p = textos or textos_reales(nombre)
    clases = FUENTES[nombre].clases if nombre in FUENTES else int(max(y_e)) + 1
    disponibles = tokenizadores_disponibles()
    elegidos = tokenizadores or list(disponibles)
    config = replace(CONFIG_REAL, **(cambios or {}))
    resultado = {"intrinsecas": {}, "filas": [], "tarea": nombre, "semillas": len(config.semillas)}
    bases = {}
    for t in elegidos:
        tarea, intr = tarea_tokenizada(nombre, disponibles[t], textos_e, y_e, textos_p, y_p, clases)
        resultado["intrinsecas"][t] = intr
        aviso(f"{t:17} fertilidad={intr['fertilidad']} desconocidas={intr['desconocidas_pct']}%")
        arqs = list(arquitecturas)
        if posiciones and t == "silabas":
            arqs += ["transformer_senoidal", "transformer_jer_pares", "transformer_jer_palabra"]
        for arq in arqs:
            corr = correr("estandar", tarea, config=replace(config, arquitectura=arq))
            resultado["filas"].append({"tokenizador": t, "arquitectura": arq, "corridas": corr})
            if t == "bpe" or (t == elegidos[0] and "bpe" not in elegidos):
                bases[arq] = corr
            aviso(f"{t:17} {arq:24} {sum(c.valor for c in corr) / len(corr):.4f}")
    # Veredicto: cada tokenizador contra BPE (el estándar de la industria) en la misma arquitectura;
    # las posiciones contra el Transformer normal con las mismas sílabas.
    silabas_tf = next((f["corridas"] for f in resultado["filas"]
                       if f["tokenizador"] == "silabas" and f["arquitectura"] == "transformer"), None)
    comps = []
    for f in resultado["filas"]:
        ref = silabas_tf if f["arquitectura"].startswith("transformer_") else bases.get(f["arquitectura"])
        if ref is not None and ref is not f["corridas"]:
            f["comparacion"] = comparar(f["corridas"], ref)
            comps.append(f["comparacion"])
    if len(comps) > 1:
        ajustar_multiples(comps)
    return resultado


def informe_tokenizadores(r: dict) -> str:
    from statistics import mean, stdev

    icono = {"mejor": "🟢 mejor", "igual": "⚪ igual", "peor": "🔴 peor", "falla": "💥 falla"}
    lineas = [f"# Tokenizadores: tus sílabas contra la industria ({r['tarea']})", "",
              "Todos en minúsculas y con el mismo vocabulario (8 000 unidades); solo cambia cómo se corta el texto.", "",
              "## Cómo cortan el texto (datos de prueba)", "",
              "| Tokenizador | Tokens por palabra | Tokens por texto | Desconocidas | Textos truncados (>256) | Palabras/s |",
              "|---|---|---|---|---|---|"]
    for t, m in r["intrinsecas"].items():
        lineas.append(f"| {t} | {m['fertilidad']} | {m['tokens_por_texto']} | {m['desconocidas_pct']} % | "
                      f"{m['truncados_pct']} % | {m['palabras_por_s']:,} |")
    lineas += ["", "Menos tokens por palabra = secuencias más cortas = modelo más rápido. Pocas desconocidas = el "
               "vocabulario cubre bien textos nuevos.", "",
               f"## Exactitud al clasificar ({r['semillas']} semillas)", "",
               "Veredicto contra **BPE** (el estándar) en la misma arquitectura; las posiciones, contra el Transformer "
               "normal con las mismas sílabas. Prueba t pareada con corrección de Benjamini-Hochberg.", "",
               "| Tokenizador | Arquitectura | Exactitud | Veredicto | Latencia (ms) | Tokens/s |", "|---|---|---|---|---|---|"]
    filas = sorted(r["filas"], key=lambda f: -mean(c.valor for c in f["corridas"]))
    for f in filas:
        v = [c.valor for c in f["corridas"]]
        sd = stdev(v) if len(v) > 1 else 0.0
        med = [c.medidas for c in f["corridas"] if c.medidas]
        lat = f"{mean(m['latencia_ms'] for m in med):.3f}" if med else "—"
        tps = f"{mean(m.get('tokens_por_s', 0) for m in med):,.0f}" if med else "—"
        ver = icono[f["comparacion"].veredicto] if "comparacion" in f else "referencia"
        lineas.append(f"| {f['tokenizador']} | {f['arquitectura']} | {mean(v) * 100:.1f} % ± {sd * 100:.1f} | {ver} | {lat} | {tps} |")
    return "\n".join(lineas) + "\n"
