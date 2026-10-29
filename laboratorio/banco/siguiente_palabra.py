"""Cuaderno 01 · «Agente PPO para segmentar texto»: en realidad predice la **siguiente palabra** con PPO.

El agente ve la palabra actual (one-hot) y elige una palabra del vocabulario. Recompensa: +1 si acierta la siguiente,
−0,1 si se equivoca, y al final de la frase +2/−1. Con ``gamma = 0,99`` cada acción se juzga por la suma descontada
de las recompensas que vienen después.

Qué se compara (mismo modelo actor de 2 capas, mismos datos: reseñas reales en español):

- ``ppo_cuaderno``: PPO tal cual (gamma 0,99, retornos normalizados, ventaja = retorno − crítico, 4 épocas, clip 0,2).
- ``ppo_arreglado``: PPO con ``gamma = 0``. La palabra siguiente no depende de lo que el agente predijo, así que
  sumar recompensas futuras solo agrega ruido: el mérito de cada predicción es su propia recompensa.
- ``supervisado``: la misma red entrenada con entropía cruzada (lo que hace cualquier modelo de lenguaje).
- ``bigramas``: contar qué palabra sigue a cada palabra (sin red; la base clásica).

Métricas en reseñas de prueba: acierto de la siguiente palabra (top-1 y top-5) y perplejidad (cuántas palabras «duda»
el modelo, en promedio; menos es mejor).

Fallos del cuaderno (anotados):
- El vocabulario tenía 14 palabras y las frases usaban otras: cualquier palabra desconocida se volvía «el», así que
  el agente casi nunca podía acertar. Aquí el vocabulario sale del texto de entrenamiento (palabras frecuentes).
- En el último paso pedía predecir la última palabra de nuevo (la misma que ya ve) en vez del fin de frase.
- Entrenaba con una frase por episodio, 10 000 episodios. Aquí se entrena por lotes de frases (el mismo algoritmo).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

FIN = "<fin>"
DESCONOCIDA = "<unk>"


def _palabras(texto: str) -> list[str]:
    return re.findall(r"[a-záéíóúüñ]+", texto.lower())


@dataclass
class Corpus:
    vocab: list[str]
    entreno: list[tuple[np.ndarray, np.ndarray]]  # (palabra actual, siguiente) por frase
    prueba: tuple[np.ndarray, np.ndarray]


def preparar(textos_e: list[str], textos_p: list[str], vocabulario: int = 2000, largo: int = 40) -> Corpus:
    cuentas = Counter(w for t in textos_e for w in _palabras(t))
    vocab = [DESCONOCIDA, FIN] + [w for w, _ in cuentas.most_common(vocabulario - 2)]
    idx = {w: i for i, w in enumerate(vocab)}

    def pares(t):
        ids = [idx.get(w, 0) for w in _palabras(t)[:largo]] + [1]
        return np.array(ids[:-1]), np.array(ids[1:])

    entreno = [p for p in map(pares, textos_e) if len(p[0])]
    xs, ys = zip(*(p for p in map(pares, textos_p) if len(p[0])))
    return Corpus(vocab, entreno, (np.concatenate(xs), np.concatenate(ys)))


def _metricas(probs: np.ndarray, y: np.ndarray) -> dict:
    top5 = np.argsort(-probs, axis=1)[:, :5]
    p = np.clip(probs[np.arange(len(y)), y], 1e-12, None)
    return {"top1": float((top5[:, 0] == y).mean()), "top5": float((top5 == y[:, None]).any(1).mean()),
            "perplejidad": float(math.exp(-np.log(p).mean()))}


def bigramas(c: Corpus, suavizado: float = 0.1) -> dict:
    v = len(c.vocab)
    m = np.full((v, v), suavizado)
    for x, y in c.entreno:
        np.add.at(m, (x, y), 1)
    m /= m.sum(1, keepdims=True)
    x, y = c.prueba
    return _metricas(m[x], y)


def entrenar_red(c: Corpus, modo: str, semilla: int = 0, epocas: int = 30, frases_por_lote: int = 64,
                 ocultas: int = 256, lr: float = 1e-3, k_epocas: int = 4, clip: float = 0.2) -> dict:
    """``modo``: "supervisado", "ppo_cuaderno" (gamma 0,99) o "ppo_arreglado" (gamma 0)."""
    import torch
    from torch import nn

    torch.manual_seed(semilla)
    rng = np.random.default_rng(semilla)
    v = len(c.vocab)
    actor = nn.Sequential(nn.Embedding(v, ocultas), nn.ReLU(), nn.Linear(ocultas, ocultas), nn.ReLU(),
                          nn.Linear(ocultas, v))  # Embedding = Linear sobre el one-hot del cuaderno
    critico = nn.Sequential(nn.Embedding(v, ocultas), nn.ReLU(), nn.Linear(ocultas, ocultas), nn.ReLU(),
                            nn.Linear(ocultas, 1))
    opt = torch.optim.Adam([*actor.parameters(), *critico.parameters()], lr=lr)
    gamma = 0.99 if modo == "ppo_cuaderno" else 0.0

    for _ in range(epocas):
        orden = rng.permutation(len(c.entreno))
        for i in range(0, len(orden), frases_por_lote):
            frases = [c.entreno[j] for j in orden[i:i + frases_por_lote]]
            x = torch.as_tensor(np.concatenate([f[0] for f in frases]))
            y = torch.as_tensor(np.concatenate([f[1] for f in frases]))
            if modo == "supervisado":
                perdida = nn.functional.cross_entropy(actor(x), y)
                opt.zero_grad(); perdida.backward(); opt.step()
                continue
            with torch.no_grad():
                dist = torch.distributions.Categorical(logits=actor(x))
                accion = dist.sample()
                logp_viejo = dist.log_prob(accion)
                acierto = (accion == y).float()
                fin = torch.zeros_like(acierto)
                fin[np.cumsum([len(f[0]) for f in frases]) - 1] = 1
                recompensa = torch.where(fin.bool(), acierto * 3 - 1, acierto * 1.1 - 0.1)  # +2/−1 al final; +1/−0,1
                retorno = torch.zeros_like(recompensa)
                acumulado = 0.0
                for t in range(len(recompensa) - 1, -1, -1):  # retorno descontado, se corta en cada fin de frase
                    acumulado = recompensa[t] + gamma * acumulado * (1 - fin[t])
                    retorno[t] = acumulado
                retorno = (retorno - retorno.mean()) / (retorno.std() + 1e-7)
            for _ in range(k_epocas):
                dist = torch.distributions.Categorical(logits=actor(x))
                valor = critico(x).squeeze(1)
                ventaja = retorno - valor.detach()
                razon = torch.exp(dist.log_prob(accion) - logp_viejo)
                politica = -torch.min(razon * ventaja, razon.clamp(1 - clip, 1 + clip) * ventaja).mean()
                perdida = politica + 0.5 * nn.functional.mse_loss(valor, retorno)
                opt.zero_grad(); perdida.backward(); opt.step()

    with torch.no_grad():
        x, y = c.prueba
        probs = torch.softmax(actor(torch.as_tensor(x)), 1).numpy()
    return _metricas(probs, y) | {"parametros": sum(p.numel() for p in actor.parameters())}


def competir(entreno: int = 5000, prueba: int = 1000, semillas=(0, 1, 2), epocas: int = 30) -> dict:
    from .datos_reales import textos_reales

    te, _, tp, _ = textos_reales("resenas_es", entreno, prueba)
    c = preparar(te, tp)
    r = {"bigramas": [bigramas(c)]}
    for modo in ("supervisado", "ppo_arreglado", "ppo_cuaderno"):
        r[modo] = [entrenar_red(c, modo, s, epocas=epocas) for s in semillas]
    r["_datos"] = {"vocabulario": len(c.vocab), "pares_entreno": sum(len(x) for x, _ in c.entreno),
                   "pares_prueba": len(c.prueba[1]), "desconocidas_prueba": float((c.prueba[1] == 0).mean())}
    return r


def informe(r: dict) -> str:
    d = r["_datos"]
    filas = ["# Cuaderno 01: «agente PPO» que predice la siguiente palabra", "",
             f"Reseñas reales en español (`resenas_es`). Vocabulario: {d['vocabulario']} palabras; "
             f"{d['pares_entreno']:,} pares para entrenar y {d['pares_prueba']:,} de prueba "
             f"({d['desconocidas_prueba']:.0%} de las palabras de prueba son desconocidas).", "",
             "| Modelo | Acierto (top-1) | Entre las 5 primeras | Perplejidad (menos es mejor) |", "|---|---|---|---|"]
    nombres = {"bigramas": "conteo de pares de palabras (clásico)", "supervisado": "misma red, supervisada (industria)",
               "ppo_arreglado": "PPO con gamma = 0 (arreglo)", "ppo_cuaderno": "PPO tal cual (gamma = 0,99)"}
    for k, nombre in nombres.items():
        vs = r[k]
        def f(m, pct=True):
            a = np.array([v[m] for v in vs])
            s = f"{a.mean():.1%}" if pct else f"{a.mean():.0f}"
            return s + (f" ± {a.std():.1%}" if pct and len(a) > 1 else (f" ± {a.std():.0f}" if len(a) > 1 else ""))
        filas.append(f"| {nombre} | {f('top1')} | {f('top5')} | {f('perplejidad', False)} |")
    return "\n".join(filas) + "\n"
