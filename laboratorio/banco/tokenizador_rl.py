"""Cuaderno 01 · «Tokenizador entrenado con RL» (celda 12), tal cual y arreglado.

Un agente recorre cada palabra letra por letra y en cada paso elige: **agregar** la letra al token actual o
**cerrar** el token y empezar otro. Aprende con Q-learning (tabla), con estado = (letra siguiente, token actual).

- ``rl_cuaderno`` (tal cual): la recompensa final es una «calidad» **al azar** (entre 5 y 10) menos 0,5 por token,
  y −0,1 por cada token cerrado. La calidad no mide nada, así que lo único que aprende es a hacer pocos tokens.
  Con letras o prefijos que no vio al entrenar (estado nuevo) elige al azar, como el cuaderno.
- ``rl_arreglado``: la recompensa mide algo real. Cada token cerrado cuesta −0,1 y, si no está en el vocabulario
  de 8 000 unidades (las más frecuentes de su propia tokenización de la ronda anterior), −1. Así el agente busca
  pocos tokens **y** que se repitan (el mismo equilibrio que buscan BPE y Unigram). Ante un estado nuevo, en vez de
  elegir al azar, sigue agregando letras mientras el token pueda llegar a ser uno del vocabulario (como WordPiece).

Se mide en la competencia de tokenizadores (``--tokenizadores``) contra BPE, Unigram, sílabas y GPT-4o.
"""

from __future__ import annotations

import random
from collections import Counter

from .tokenizadores import _PALABRA, _TROZOS, INICIO, Tokenizador

AGREGAR, CERRAR = 0, 1


class TokenizadorRL(Tokenizador):
    def __init__(self, arreglado: bool, episodios: int = 60_000, rondas: int = 3, vocabulario: int = 8000,
                 semilla: int = 0) -> None:
        self.arreglado, self.episodios, self.rondas, self.tamano = arreglado, episodios, rondas, vocabulario
        self.semilla = semilla
        self.nombre = "rl_arreglado" if arreglado else "rl_cuaderno"
        self.descripcion = ("tokenizador con RL, recompensa real y vocabulario (arreglo)" if arreglado
                            else "tokenizador con RL del cuaderno 01 (recompensa al azar), tal cual")
        self.q: dict = {}
        self.vocab: set[str] = set()
        self.prefijos: set[str] = set()

    # ------------------------------------------------------------ política
    def _elegir(self, estado, epsilon: float, rng: random.Random) -> int:
        if rng.random() < epsilon:
            return rng.randrange(2)
        valores = self.q.get(estado)
        if valores is None:
            if not self.arreglado:
                return rng.randrange(2)
            letra, actual = estado
            return AGREGAR if letra is not None and (actual + letra) in self.prefijos else CERRAR
        return AGREGAR if valores[AGREGAR] >= valores[CERRAR] else CERRAR

    def _cortar(self, palabra: str, epsilon: float, rng: random.Random, aprender: bool):
        i, actual, tokens, total = 0, "", [], 0.0
        for _ in range(4 * len(palabra) + 4):  # tope de pasos (el cuaderno no tenía y podía no terminar)
            letra = palabra[i] if i < len(palabra) else None
            estado = (letra, actual)
            accion = self._elegir(estado, epsilon, rng) if letra is not None else CERRAR
            recompensa, fin = 0.0, False
            if accion == AGREGAR:
                actual += letra
                i += 1
            elif actual:
                tokens.append(actual)
                recompensa -= 0.1
                if self.arreglado and actual not in self.vocab:
                    recompensa -= 1.0
                actual = ""
            else:
                recompensa -= 1.0  # cerrar un token vacío
            if i >= len(palabra) and not actual:
                fin = True
                if not self.arreglado:
                    recompensa += rng.uniform(0.5, 1.0) * 10 - 0.5 * len(tokens)  # «calidad» simulada
            if aprender:
                siguiente = (palabra[i] if i < len(palabra) else None, actual)
                q = self.q.setdefault(estado, [0.0, 0.0])
                futuro = 0.0 if fin else max(self.q.get(siguiente, [0.0, 0.0]))
                q[accion] += 0.1 * (recompensa + 0.9 * futuro - q[accion])
            total += recompensa
            if fin:
                break
        if actual:
            tokens.append(actual)
        return tokens

    def _actualizar_vocab(self, palabras: list[str]) -> None:
        rng = random.Random(self.semilla)
        cuentas = Counter(t for w in palabras for t in self._cortar(w, 0.0, rng, False))
        cuentas.update(set("".join(palabras)))  # las letras sueltas siempre están (nada queda sin cubrir)
        self.vocab = {t for t, _ in cuentas.most_common(self.tamano)}
        self.prefijos = {t[:k] for t in self.vocab for k in range(1, len(t) + 1)}

    def entrenar(self, textos: list[str]) -> None:
        rng = random.Random(self.semilla)
        palabras = [w for t in textos for w in _PALABRA.findall(t.lower())]
        self.q = {}
        rondas = self.rondas if self.arreglado else 1
        if self.arreglado:
            frecuentes = Counter(palabras)
            self.vocab = set(list(set("".join(palabras)))) | {w for w, _ in frecuentes.most_common(self.tamano // 2)}
            self.prefijos = {t[:k] for t in self.vocab for k in range(1, len(t) + 1)}
        for _ in range(rondas):
            epsilon = 0.9
            for e in range(self.episodios // rondas):
                self._cortar(rng.choice(palabras), epsilon, rng, True)
                epsilon = max(0.01, epsilon * 0.9995)
            if self.arreglado:
                self._actualizar_vocab(palabras[:50_000])

    def tokens(self, texto: str) -> list[str]:
        rng = random.Random(self.semilla)
        out = []
        for trozo in _TROZOS.findall(texto.lower()):
            partes = self._cortar(trozo, 0.0, rng, False) if _PALABRA.fullmatch(trozo) else [trozo]
            partes = partes or [trozo]
            out.append(INICIO + partes[0])
            out.extend(partes[1:])
        return out
