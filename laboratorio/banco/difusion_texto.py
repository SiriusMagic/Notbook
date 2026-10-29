"""Cuaderno 01 · Difusión de texto («ImprovedDiffusionTextModel» y «TextusDiffusioRestituo», el mismo modelo).

El cuaderno toma frases, les suma ruido gaussiano a los embeddings y entrena un Transformer para devolver las
palabras originales. Para generar, parte de una frase llena de [MASK] y en 50 pasos predice todas las palabras.

Fallos de diseño (anotados en el código):
1. **El ruido es casi nulo.** La varianza máxima es 0,01 (desviación 0,1) sobre embeddings de norma ≈ 11: el
   modelo nunca ve una frase realmente dañada y aprende a copiar.
2. **Los embeddings no aprenden**: se calculan sin gradiente (``torch.no_grad``), así que quedan al azar.
3. **Genera desde [MASK], algo que nunca vio al entrenar** (entrenó con ruido gaussiano, no con máscaras).
4. Con 4 frases de ejemplo solo puede memorizarlas.

Qué se compara (mismos datos: reseñas reales en español, mismo Transformer de 2 capas, mismos tokens):

- ``difusion_cuaderno``: tal cual (ruido gaussiano 0,0001–0,01, embeddings fijos), muestreo desde [MASK].
- ``difusion_mascaras``: el arreglo, como la difusión de texto de la industria (MDLM, Sahoo et al. 2024; LLaDA, 2025):
  el «ruido» es tapar con [MASK] una fracción t de las palabras (t al azar entre 0 y 1) y el modelo aprende a
  destaparlas. Para generar, parte de todo tapado y destapa poco a poco.
- ``autorregresivo``: un Transformer que escribe palabra por palabra (como GPT), del mismo tamaño.

Métricas en reseñas de prueba:
- **Relleno**: se tapa el 15 % o el 50 % de las palabras y se mide cuántas acierta (el autorregresivo no puede
  rellenar huecos en medio de la frase, así que no participa).
- **Perplejidad** (cuántas palabras «duda» en promedio; menos es mejor): exacta para el autorregresivo; para la
  difusión con máscaras es una **cota superior** (la de MDLM). La difusión del cuaderno no define una probabilidad.
"""

from __future__ import annotations

import math

import numpy as np

PAD, MASK, BOS = 0, 1, 2


def preparar(textos_e: list[str], textos_p: list[str], vocabulario: int = 4000, largo: int = 32):
    """BPE entrenado con las reseñas; secuencias de ``largo`` tokens: [BOS] + texto, relleno con PAD."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    tok.decoder = decoders.Metaspace()
    tok.train_from_iterator((t.lower() for t in textos_e), trainer=trainers.BpeTrainer(
        vocab_size=vocabulario, special_tokens=["[PAD]", "[MASK]", "[BOS]", "[UNK]"], show_progress=False))

    def codificar(textos):
        m = np.zeros((len(textos), largo), dtype=np.int64)
        for i, t in enumerate(textos):
            ids = [BOS] + tok.encode(t.lower()).ids[: largo - 1]
            m[i, : len(ids)] = ids
        return m

    return tok, codificar(textos_e), codificar(textos_p)


def _modelo(v: int, largo: int, d: int = 128, capas: int = 2, cabezas: int = 4, pasos_t: int = 50):
    import torch
    from torch import nn

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(v, d, padding_idx=PAD)
            self.pos = nn.Parameter(torch.randn(1, largo, d) * 0.02)
            self.tiempo = nn.Embedding(pasos_t + 1, d)
            capa = nn.TransformerEncoderLayer(d, cabezas, 4 * d, dropout=0.1, batch_first=True)
            self.cuerpo = nn.TransformerEncoder(capa, capas)
            self.norma = nn.LayerNorm(d)
            self.salida = nn.Linear(d, v)

        def forward(self, x_emb, t=None, causal=False, relleno=None):
            h = x_emb + self.pos[:, : x_emb.size(1)]
            if t is not None:
                h = h + self.tiempo(t).unsqueeze(1)
            mascara = nn.Transformer.generate_square_subsequent_mask(h.size(1), device=h.device) if causal else None
            return self.salida(self.norma(self.cuerpo(h, mask=mascara, is_causal=causal, src_key_padding_mask=relleno)))

    return Transformer()


def entrenar(modo: str, xe: np.ndarray, v: int, semilla: int = 0, pasos: int = 3000, lote: int = 64,
             lr: float = 3e-4, pasos_t: int = 50, dispositivo: str = "auto"):
    """``modo``: "difusion_cuaderno", "difusion_mascaras" o "autorregresivo". ``dispositivo``: auto, cpu o cuda."""
    import torch
    from torch import nn

    torch.manual_seed(semilla)
    rng = np.random.default_rng(semilla)
    largo = xe.shape[1]
    if dispositivo == "auto":
        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(dispositivo)
    m = _modelo(v, largo, pasos_t=pasos_t).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for _ in range(pasos):
        x = torch.as_tensor(xe[rng.integers(0, len(xe), lote)], device=dev)
        real = x != PAD
        if modo == "autorregresivo":
            logits = m(m.emb(x[:, :-1]), causal=True)
            perdida = nn.functional.cross_entropy(logits.reshape(-1, v), x[:, 1:].reshape(-1), ignore_index=PAD)
        elif modo == "difusion_cuaderno":
            t = torch.randint(0, pasos_t, (lote,), device=dev)
            with torch.no_grad():  # tal cual: los embeddings no reciben gradiente
                e = m.emb(x)
            beta = 0.0001 + t.float() * (0.01 - 0.0001) / (pasos_t - 1)
            ruidoso = e + torch.randn_like(e) * beta.sqrt().view(-1, 1, 1)
            logits = m(ruidoso, t, relleno=~real)
            perdida = nn.functional.cross_entropy(logits.reshape(-1, v), x.reshape(-1), ignore_index=PAD)
        else:  # difusión con máscaras (MDLM): tapar una fracción t, pérdida pesada por 1/t sobre lo tapado
            t = torch.rand(lote, device=dev).clamp(min=1e-3)
            tapar = (torch.rand(x.shape, device=dev) < t.view(-1, 1)) & real
            tapar[:, 0] = False  # el [BOS] queda visible
            xr = x.masked_fill(tapar, MASK)
            logits = m(m.emb(xr), (t * pasos_t).long())
            ce = nn.functional.cross_entropy(logits.reshape(-1, v), x.reshape(-1), reduction="none").view(x.shape)
            perdida = ((ce * tapar).sum(1) / t).sum() / real.sum()
        opt.zero_grad()
        perdida.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    return m


def rellenar(m, modo: str, xp: np.ndarray, fraccion: float, semilla: int = 0, pasos_t: int = 50) -> float:
    """Tapa ``fraccion`` de los tokens reales (no el [BOS]) y mide cuántos acierta el modelo de una pasada."""
    import torch

    dev = next(m.parameters()).device
    g = torch.Generator().manual_seed(semilla)
    x = torch.as_tensor(xp)
    real = x != PAD
    tapar = (torch.rand(x.shape, generator=g) < fraccion) & real
    x, real, tapar = x.to(dev), real.to(dev), tapar.to(dev)
    tapar[:, 0] = False
    xr = x.masked_fill(tapar, MASK)
    m.eval()
    with torch.no_grad():
        t = torch.full((len(x),), (pasos_t - 1) if modo == "difusion_cuaderno" else int(fraccion * pasos_t), device=dev)
        pred = m(m.emb(xr), t, relleno=~real if modo == "difusion_cuaderno" else None).argmax(-1)
    return float((pred[tapar] == x[tapar]).float().mean())


def perplejidad(m, modo: str, xp: np.ndarray, semilla: int = 0, muestras: int = 8, pasos_t: int = 50) -> float | None:
    import torch
    from torch import nn

    dev = next(m.parameters()).device
    x = torch.as_tensor(xp, device=dev)
    real = x != PAD
    v = m.salida.out_features
    m.eval()
    with torch.no_grad():
        if modo == "autorregresivo":
            logits = m(m.emb(x[:, :-1]), causal=True)
            nll = nn.functional.cross_entropy(logits.reshape(-1, v), x[:, 1:].reshape(-1), ignore_index=PAD,
                                              reduction="sum")
            return math.exp(float(nll) / float(real[:, 1:].sum()))
        if modo != "difusion_mascaras":
            return None
        g = torch.Generator().manual_seed(semilla)
        total = 0.0
        for _ in range(muestras):  # cota de MDLM: E_t [ (1/t) · Σ CE en los tapados ]
            t = torch.rand(len(x), generator=g).clamp(min=1e-3).to(dev)
            tapar = (torch.rand(x.shape, generator=g).to(dev) < t.view(-1, 1)) & real
            tapar[:, 0] = False
            logits = m(m.emb(x.masked_fill(tapar, MASK)), (t * pasos_t).long())
            ce = nn.functional.cross_entropy(logits.reshape(-1, v), x.reshape(-1), reduction="none").view(x.shape)
            total += float(((ce * tapar).sum(1) / t).sum())
        return math.exp(total / muestras / float(real[:, 1:].sum()))


def generar(m, modo: str, tok, n: int = 3, largo: int = 20, semilla: int = 0, pasos_t: int = 50) -> list[str]:
    """Ejemplos de texto nuevo (para mirarlos; no entran en la puntuación)."""
    import torch

    torch.manual_seed(semilla)
    dev = next(m.parameters()).device
    m.eval()
    with torch.no_grad():
        if modo == "autorregresivo":
            x = torch.full((n, 1), BOS, device=dev)
            for _ in range(largo - 1):
                p = torch.softmax(m(m.emb(x), causal=True)[:, -1], -1)
                x = torch.cat([x, torch.multinomial(p, 1)], 1)
        else:
            x = torch.full((n, largo), MASK, device=dev)
            x[:, 0] = BOS
            for paso in range(pasos_t, 0, -1):
                t = torch.full((n,), paso if modo == "difusion_mascaras" else paso - 1, device=dev)
                logits = m(m.emb(x), t)
                if modo == "difusion_cuaderno":  # tal cual: reemplaza todo por la palabra más probable
                    x = logits.argmax(-1)
                    continue
                tapado = x == MASK
                nuevo = torch.distributions.Categorical(logits=logits).sample()
                destapar = tapado & (torch.rand(x.shape, device=dev) < 1 / paso)  # destapa ≈ 1/paso de lo que queda
                x = torch.where(destapar, nuevo, x)
    return [tok.decode([i for i in fila.tolist() if i > 3]) for fila in x.cpu()]


def competir(entreno: int = 10000, prueba: int = 1000, semillas=(0, 1, 2), pasos: int = 3000,
             dispositivo: str = "auto") -> dict:
    from .datos_reales import textos_reales

    te, _, tp, _ = textos_reales("resenas_es", entreno, prueba)
    tok, xe, xp = preparar(te, tp)
    v = tok.get_vocab_size()
    r = {"_datos": {"entreno": len(xe), "prueba": len(xp), "vocabulario": v, "largo": xe.shape[1]}}
    for modo in ("difusion_cuaderno", "difusion_mascaras", "autorregresivo"):
        filas = []
        for s in semillas:
            m = entrenar(modo, xe, v, s, pasos=pasos, dispositivo=dispositivo)
            fila = {"perplejidad": perplejidad(m, modo, xp, s),
                    "parametros": sum(p.numel() for p in m.parameters())}
            if modo != "autorregresivo":
                fila |= {"relleno_15": rellenar(m, modo, xp, 0.15, s), "relleno_50": rellenar(m, modo, xp, 0.5, s)}
            if s == semillas[0]:
                fila["ejemplos"] = generar(m, modo, tok)
            filas.append(fila)
        r[modo] = filas
    return r


def informe(r: dict) -> str:
    d = r["_datos"]
    nombres = {"difusion_cuaderno": "difusión del cuaderno (tal cual)",
               "difusion_mascaras": "difusión con máscaras (arreglo, como MDLM/LLaDA)",
               "autorregresivo": "Transformer autorregresivo (como GPT)"}

    def f(vals, pct):
        vals = [x for x in vals if x is not None]
        if not vals:
            return "—"
        a = np.array(vals)
        return (f"{a.mean():.1%} ± {a.std():.1%}" if pct else f"{a.mean():.0f} ± {a.std():.0f}") if len(a) > 1 else \
            (f"{a.mean():.1%}" if pct else f"{a.mean():.0f}")

    lineas = ["# Cuaderno 01: difusión de texto", "",
              f"Reseñas reales en español: {d['entreno']:,} para entrenar y {d['prueba']:,} de prueba, cortadas en "
              f"{d['largo']} tokens (BPE de {d['vocabulario']:,}). Mismo Transformer de 2 capas para todos.", "",
              "| Modelo | Rellena 15 % tapado | Rellena 50 % tapado | Perplejidad (menos es mejor) | Parámetros |",
              "|---|---|---|---|---|"]
    for k, nombre in nombres.items():
        fs = r[k]
        lineas.append(f"| {nombre} | {f([x.get('relleno_15') for x in fs], True)} | "
                      f"{f([x.get('relleno_50') for x in fs], True)} | {f([x['perplejidad'] for x in fs], False)} | "
                      f"{fs[0]['parametros']:,} |")
    lineas += ["", "La perplejidad de la difusión con máscaras es una cota superior (la real puede ser menor). "
               "La difusión del cuaderno no define una probabilidad, así que no tiene perplejidad.", "",
               "## Ejemplos generados (semilla 0)", ""]
    for k, nombre in nombres.items():
        lineas.append(f"**{nombre}**:")
        lineas += [f"- «{t}»" for t in r[k][0]["ejemplos"]]
        lineas.append("")
    return "\n".join(lineas)
