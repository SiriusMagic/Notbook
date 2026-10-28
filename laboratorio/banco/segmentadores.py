"""Competencia de segmentadores de sílabas (cuaderno 01).

| Segmentador | Qué es |
|---|---|
| ``reglas`` | reglas del español (``laboratorio/silabas.py``) |
| ``cuaderno`` | la heurística original del cuaderno 01 (``SimpleSyllableTokenizer``) |
| ``pyphen`` | diccionario de guionado de LibreOffice/Hunspell (lo que usa la industria para cortar palabras) |
| ``bilstm`` | la red del cuaderno 01 (celda 16): embeddings de letras + LSTM bidireccional, tal cual |
| ``transformer`` | Transformer de caracteres pequeño (referencia neuronal moderna) |

Las redes aprenden de palabras reales de las reseñas, etiquetadas con las
reglas (no hay un silabeo «humano» de miles de palabras). Todas se miden con la
lista de referencia ``silabas_referencia.py`` (128 palabras silabeadas a mano
según la RAE), que **no** se usa para entrenar.

Métricas:
- **palabra exacta**: % de palabras con todos los cortes bien;
- **F1 de cortes**: equilibrio entre cortes que sobran y cortes que faltan;
- **velocidad** (palabras/s) y, en las redes, parámetros y latencia.
"""

from __future__ import annotations

import re
import time
from statistics import mean, stdev

import torch
import torch.nn.functional as F
from torch import nn

from ..silabas import silabear
from .silabas_referencia import REFERENCIA

LETRAS = list("abcdefghijklmnñopqrstuvwxyzáéíóúü")  # el vocabulario de la celda 16
INDICE = {c: i + 2 for i, c in enumerate(LETRAS)}  # 0 = relleno, 1 = otra letra
LARGO = 24


def cortes(silabeo: str) -> list[int]:
    """'ca-sa' → [0, 1, 0]: 1 si hay corte después de esa letra (sin contar la última)."""
    partes = silabeo.lower().split("-")
    salida: list[int] = []
    for p in partes[:-1]:
        salida += [0] * (len(p) - 1) + [1]
    salida += [0] * (len(partes[-1]) - 1)
    return salida


def unir(palabra: str, marcas: list[int]) -> str:
    return "".join(c + ("-" if i < len(marcas) and marcas[i] else "") for i, c in enumerate(palabra))


def _codificar(palabras: list[str]) -> torch.Tensor:
    m = torch.zeros(len(palabras), LARGO, dtype=torch.long)
    for i, w in enumerate(palabras):
        for j, c in enumerate(w.lower()[:LARGO]):
            m[i, j] = INDICE.get(c, 1)
    return m


def _etiquetas(silabeos: list[str]) -> torch.Tensor:
    m = torch.zeros(len(silabeos), LARGO)
    for i, s in enumerate(silabeos):
        c = cortes(s)[:LARGO]
        m[i, : len(c)] = torch.tensor(c, dtype=torch.float)
    return m


# ---------------------------------------------------------------- Redes
class SegmentadorBiLSTM(nn.Module):
    """La red de la celda 16 del cuaderno 01, sin cambios de arquitectura."""

    def __init__(self, embed_dim: int = 32, hidden_dim: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(len(LETRAS) + 2, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(self.embedding(x))
        return self.classifier(out).squeeze(-1)


class SegmentadorTransformer(nn.Module):
    """Transformer de caracteres: cada letra mira a todas las demás de la palabra."""

    def __init__(self, dim: int = 64, capas: int = 2) -> None:
        super().__init__()
        self.emb = nn.Embedding(len(LETRAS) + 2, dim, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, LARGO, dim))
        capa = nn.TransformerEncoderLayer(dim, 4, 128, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(capa, capas, enable_nested_tensor=False)
        self.salida = nn.Linear(dim, 1)

    def forward(self, x):
        h = self.enc(self.emb(x) + self.pos[:, : x.size(1)], src_key_padding_mask=x == 0)
        return self.salida(h).squeeze(-1)


REDES = {"bilstm": SegmentadorBiLSTM, "transformer": SegmentadorTransformer}


def _predecir(modelo: nn.Module, palabras: list[str]) -> list[str]:
    modelo.eval()
    with torch.no_grad():
        p = torch.sigmoid(modelo(_codificar(palabras).to(next(modelo.parameters()).device))).cpu()
    return [unir(w, (p[i, : max(len(w) - 1, 0)] > 0.5).int().tolist()) for i, w in enumerate(palabras)]


def entrenar_red(nombre: str, palabras: list[str], semilla: int, epocas: int = 60, lote: int = 128,
                 paciencia: int = 6, dispositivo: str = "auto") -> tuple[nn.Module, dict]:
    """Entrena con etiquetas de las reglas; separa 10 % para parar a tiempo."""
    disp = torch.device("cuda" if dispositivo == "auto" and torch.cuda.is_available() else
                        ("cpu" if dispositivo == "auto" else dispositivo))
    torch.manual_seed(semilla)
    gen = torch.Generator().manual_seed(semilla)
    palabras = [w for w in palabras if 1 < len(w) <= LARGO]
    orden = torch.randperm(len(palabras), generator=torch.Generator().manual_seed(1234)).tolist()
    n_val = max(1, len(palabras) // 10)
    val = [palabras[i] for i in orden[:n_val]]
    ent = [palabras[i] for i in orden[n_val:]]
    x, y = _codificar(ent).to(disp), _etiquetas([silabear(w) for w in ent]).to(disp)
    modelo = REDES[nombre]().to(disp)
    opt = torch.optim.Adam(modelo.parameters(), lr=0.003)
    oro_val = [silabear(w).lower() for w in val]
    mejor, mejor_estado, sin_mejora, epoca = -1.0, None, 0, 0
    inicio = time.perf_counter()
    for epoca in range(epocas):
        modelo.train()
        for idx in torch.randperm(len(ent), generator=gen).split(lote):
            idx = idx.to(disp)
            mascara = (x[idx] != 0).float()
            loss = (F.binary_cross_entropy_with_logits(modelo(x[idx]), y[idx], reduction="none") * mascara).sum() / mascara.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        acierto = mean(p == o for p, o in zip(_predecir(modelo, val), oro_val))
        if acierto > mejor:
            mejor, sin_mejora = acierto, 0
            mejor_estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    modelo.load_state_dict(mejor_estado)
    return modelo, {"segundos_entrenar": round(time.perf_counter() - inicio, 2), "epocas": epoca + 1,
                    "parametros": sum(p.numel() for p in modelo.parameters()), "validacion": round(mejor, 4)}


# ---------------------------------------------------------------- Medición
def evaluar(silabear_fn, referencia: dict[str, str] = REFERENCIA) -> dict:
    palabras = list(referencia)
    t0 = time.perf_counter()
    predichas = [silabear_fn(w).lower() for w in palabras]
    segundos = time.perf_counter() - t0
    exactas = sum(p == referencia[w].lower() for p, w in zip(predichas, palabras))
    vp = fp = fn = 0
    for p, w in zip(predichas, palabras):
        a, b = cortes(p), cortes(referencia[w])
        if len(a) != len(b):  # la predicción cambió letras (no debería): cuenta todo como error
            fn += sum(b)
            fp += sum(a)
            continue
        vp += sum(1 for i, j in zip(a, b) if i and j)
        fp += sum(1 for i, j in zip(a, b) if i and not j)
        fn += sum(1 for i, j in zip(a, b) if j and not i)
    precision = vp / (vp + fp) if vp + fp else 0.0
    recall = vp / (vp + fn) if vp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    errores = [(w, p, referencia[w]) for p, w in zip(predichas, palabras) if p != referencia[w].lower()]
    return {"exactas": exactas / len(palabras), "f1": f1, "palabras_por_s": len(palabras) / max(segundos, 1e-9),
            "errores": errores}


def segmentadores_fijos() -> dict:
    from .tokenizadores import _silabear_cuaderno

    fijos = {"reglas": silabear, "cuaderno": lambda w: "-".join(_silabear_cuaderno(w.lower()))}
    try:
        import pyphen

        dic = pyphen.Pyphen(lang="es_ES")
        fijos["pyphen"] = lambda w: dic.inserted(w.lower())
    except ImportError:
        pass
    return fijos


def palabras_de(textos: list[str], excluir=REFERENCIA) -> list[str]:
    """Palabras distintas (solo letras) de unos textos, sin las de la lista de referencia."""
    vistas = {w for t in textos for w in re.findall(r"[a-záéíóúüñ]+", t.lower())}
    return sorted(vistas - {w.lower() for w in excluir})


def competir_segmentadores(palabras: list[str], semillas=(0, 1, 2), redes=("bilstm", "transformer"),
                           aviso=print, **entreno) -> dict:
    filas = []
    for nombre, fn in segmentadores_fijos().items():
        r = evaluar(fn)
        filas.append({"nombre": nombre, "tipo": "fijo", "exactas": [r["exactas"]], "f1": [r["f1"]],
                      "palabras_por_s": r["palabras_por_s"], "errores": r["errores"]})
        aviso(f"{nombre:12} exactas={r['exactas']:.3f}")
    for nombre in redes:
        exactas, f1s, info, errores, vel = [], [], [], [], []
        for s in semillas:
            modelo, datos = entrenar_red(nombre, palabras, s, **entreno)
            r = evaluar(lambda w, m=modelo: _predecir(m, [w])[0])  # una palabra a la vez = latencia real
            exactas.append(r["exactas"])
            f1s.append(r["f1"])
            vel.append(r["palabras_por_s"])
            info.append(datos)
            errores = r["errores"]
        filas.append({"nombre": nombre, "tipo": "red", "exactas": exactas, "f1": f1s, "palabras_por_s": mean(vel),
                      "errores": errores, "parametros": info[0]["parametros"],
                      "segundos_entrenar": mean(d["segundos_entrenar"] for d in info)})
        aviso(f"{nombre:12} exactas={mean(exactas):.3f}")
    return {"filas": filas, "palabras_entreno": len(palabras), "semillas": len(semillas)}


def informe_segmentadores(r: dict) -> str:
    def fmt(v):
        return f"{mean(v) * 100:.1f} %" + (f" ± {stdev(v) * 100:.1f}" if len(v) > 1 else "")

    lineas = ["# Segmentadores de sílabas", "",
              f"Medidos con {len(REFERENCIA)} palabras silabeadas a mano según la RAE (`silabas_referencia.py`). "
              f"Las redes aprendieron de {r['palabras_entreno']:,} palabras reales de reseñas etiquetadas con las "
              f"reglas ({r['semillas']} semillas). Aviso: la lista de referencia y las reglas son del mismo autor.", "",
              "| Segmentador | Palabra exacta | F1 de cortes | Palabras/s | Parámetros | Seg. entrenar |",
              "|---|---|---|---|---|---|"]
    for f in sorted(r["filas"], key=lambda f: -mean(f["exactas"])):
        if f["tipo"] == "red":
            extra = f"{f['parametros']:,} | {f['segundos_entrenar']:.1f}"
        else:
            extra = "— (reglas) | —"
        lineas.append(f"| {f['nombre']} | {fmt(f['exactas'])} | {fmt(f['f1'])} | {f['palabras_por_s']:,.0f} | {extra} |")
    lineas += ["", "## Ejemplos de errores", ""]
    for f in r["filas"]:
        if f["errores"]:
            muestra = ", ".join(f"{w}: `{p}` (es `{o}`)" for w, p, o in f["errores"][:6])
            lineas.append(f"- **{f['nombre']}** ({len(f['errores'])} errores): {muestra}")
        else:
            lineas.append(f"- **{f['nombre']}**: sin errores.")
    return "\n".join(lineas) + "\n"


__all__ = ["competir_segmentadores", "informe_segmentadores", "palabras_de", "evaluar", "cortes", "unir"]
