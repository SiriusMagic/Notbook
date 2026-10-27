"""Separación en sílabas del español y tokenizador por sílabas.

Viene de los cuadernos «Segmentador de sílabas» y «Tokenizador por sílabas»
(01_texto_y_lenguaje). La versión del cuaderno usaba una heurística que fallaba
con diptongos, hiatos, «qu/gu» y grupos como «ns-t»; aquí se aplican las reglas
de la ortografía del español:

- «ch», «ll», «rr» y «qu/gu» (ante e/i) cuentan como una sola consonante.
- Vocales juntas: abierta + abierta, o vocal cerrada con tilde, forman hiato
  (se separan); en otro caso forman diptongo o triptongo (no se separan).
- Entre vocales, una consonante va con la sílaba siguiente; dos se separan,
  salvo grupos como «bl», «pr», «tr» que van juntos; con tres o cuatro se
  separa antes del grupo inseparable.
"""

from __future__ import annotations

import re
from collections import Counter

ABIERTAS = set("aeoáéóÁÉÓAEO") | set("íúÍÚ")  # vocal cerrada con tilde se comporta como abierta
CERRADAS = set("iuüIUÜ")
VOCALES = ABIERTAS | CERRADAS
GRUPOS_INSEPARABLES = {
    "pl", "pr", "bl", "br", "fl", "fr", "cl", "cr", "gl", "gr", "kl", "kr", "dr", "tr",
}
_PALABRA = re.compile(r"[a-záéíóúüñA-ZÁÉÍÓÚÜÑ]+")


def _unidades(palabra: str) -> list[tuple[str, str]]:
    """Divide la palabra en unidades (texto, tipo) con tipo 'V' (vocal) o 'C' (consonante)."""
    p = palabra
    low = palabra.lower()
    out: list[tuple[str, str]] = []
    i = 0
    while i < len(p):
        par = low[i : i + 2]
        sig = low[i + 2] if i + 2 < len(p) else ""
        if par in ("ch", "ll", "rr"):
            out.append((p[i : i + 2], "C"))
            i += 2
        elif par in ("qu", "gu") and sig in ("e", "i", "é", "í"):
            out.append((p[i : i + 2], "C"))  # la «u» no suena: que, gui
            i += 2
        elif low[i] == "y":
            # «y» es vocal al final de palabra o entre consonantes (rey, hoy); si no, consonante.
            antes_vocal = i + 1 < len(p) and low[i + 1] in {v.lower() for v in VOCALES}
            out.append((p[i], "C" if antes_vocal or i == 0 else "V"))
            i += 1
        else:
            out.append((p[i], "V" if p[i] in VOCALES else "C"))
            i += 1
    return out


def _es_hiato(a: str, b: str) -> bool:
    a, b = a.lower(), b.lower()
    a = "i" if a == "y" else a
    b = "i" if b == "y" else b
    abiertas = {v.lower() for v in ABIERTAS}
    return (a in abiertas and b in abiertas) or (a == b and a in "iu")


def silabas(palabra: str) -> list[str]:
    """Separa una palabra en sílabas: ``silabas("chocolate") -> ['cho', 'co', 'la', 'te']``."""
    if not palabra:
        return []
    u = _unidades(palabra)
    # Núcleos: índices de vocales agrupadas (diptongos juntos, hiatos separados).
    nucleos: list[list[int]] = []
    for i, (t, tipo) in enumerate(u):
        if tipo != "V":
            continue
        if nucleos and nucleos[-1][-1] == i - 1 and not _es_hiato(u[i - 1][0], t):
            nucleos[-1].append(i)
        else:
            nucleos.append([i])
    if not nucleos:
        return [palabra]

    cortes: list[int] = []  # índice de unidad donde empieza cada sílaba (salvo la primera)
    for prev, nxt in zip(nucleos, nucleos[1:]):
        ini, fin = prev[-1] + 1, nxt[0]  # consonantes entre ambos núcleos: u[ini:fin]
        cons = [t.lower() for t, _ in u[ini:fin]]
        n = len(cons)
        if n == 0:
            corte = fin  # hiato
        elif n == 1:
            corte = ini
        elif n == 2:
            corte = ini if cons[0] + cons[1] in GRUPOS_INSEPARABLES else ini + 1
        else:
            corte = fin - 2 if cons[-2] + cons[-1] in GRUPOS_INSEPARABLES else fin - 1
        cortes.append(corte)

    partes, desde = [], 0
    for c in cortes:
        partes.append("".join(t for t, _ in u[desde:c]))
        desde = c
    partes.append("".join(t for t, _ in u[desde:]))
    return [s for s in partes if s]


def silabear(palabra: str, separador: str = "-") -> str:
    """``silabear("instante") -> 'ins-tan-te'``."""
    return separador.join(silabas(palabra))


def silabas_a_vector(palabra: str, silabeo: str) -> tuple[str, list[int]]:
    """Marca con 1 la letra tras la cual hay un corte de sílaba.

    ``silabas_a_vector("casa", "ca-sa") -> ('casa', [0, 1, 0])``. Sirve como
    etiqueta para entrenar un modelo que aprenda a silabear.
    """
    palabra = palabra.replace(" ", "")
    vector = [0] * max(len(palabra) - 1, 0)
    indice = 0
    for silaba in silabeo.split("-")[:-1]:
        indice += len(silaba)
        if 0 < indice < len(palabra):
            vector[indice - 1] = 1
    return palabra, vector


class TokenizadorSilabas:
    """Convierte texto en sílabas y en números (ids), para alimentar un modelo."""

    ESPECIALES = {"<pad>": 0, "<unk>": 1}

    def __init__(self) -> None:
        self.vocab: dict[str, int] = dict(self.ESPECIALES)
        self.inverso: dict[int, str] = {i: s for s, i in self.vocab.items()}

    @staticmethod
    def tokenizar(texto: str) -> list[str]:
        """Sílabas en minúsculas; los signos de puntuación quedan como token propio."""
        tokens: list[str] = []
        for trozo in re.findall(r"\w+|[^\w\s]", texto.lower()):
            tokens.extend(silabas(trozo) if _PALABRA.fullmatch(trozo) else [trozo])
        return tokens

    def construir_vocabulario(self, textos: list[str], frecuencia_minima: int = 1) -> None:
        cuentas = Counter(s for t in textos for s in self.tokenizar(t))
        self.vocab = dict(self.ESPECIALES)
        for silaba in sorted(s for s, n in cuentas.items() if n >= frecuencia_minima):
            self.vocab.setdefault(silaba, len(self.vocab))
        self.inverso = {i: s for s, i in self.vocab.items()}

    def a_ids(self, texto: str) -> list[int]:
        return [self.vocab.get(s, self.vocab["<unk>"]) for s in self.tokenizar(texto)]

    def a_silabas(self, ids: list[int]) -> list[str]:
        return [self.inverso.get(i, "<unk>") for i in ids]
