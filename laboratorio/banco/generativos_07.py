"""Cuaderno 07 · Parte generativa: D-ReG (texto) y HCMG (imágenes condicionadas).

**D-ReG** («Generador de texto recurrente dual»): en cada paso, una corriente de «percepción» (atención +
red feed-forward) y una de «conceptualización» (GRU) se influyen con «unidades de interacción cruzada» (atención),
y la percepción predice la palabra siguiente.
- Fallo: todas las atenciones reciben **un solo elemento** (secuencia de largo 1). Con un elemento, el softmax vale
  1 y la atención solo copia: no elige nada. La «percepción» no ve las palabras anteriores; solo la GRU recuerda.
- Fallo: el modelo solo generaba (con argmax); no tenía forma de entrenarse (sin pérdida ni *teacher forcing*).
- Aquí se entrena prediciendo la palabra siguiente. El codificador de 6 capas del prompt no se usa (no hay prompt:
  se modela el texto completo).
- Arreglo: la atención de la percepción mira las últimas palabras (ventana de 16), como en un Transformer.
- Rivales del mismo tamaño: GRU, LSTM y Transformer causal (como GPT).

**HCMG** («Generador condicionado multimodal jerárquico»): un codificador-decodificador convolucional donde cada
bloque normaliza y luego reescala sus activaciones según dos vectores de condición (imagen/ID y texto), uno tras
otro (como AdaIN o SPADE).
- Tarea: a partir de un dígito **muy borroso**, dibujar el dígito nítido de la **clase** pedida (condición «ID») y
  en el **color** pedido (condición «texto»). Ninguna de las dos se puede adivinar del borrón: el modelo tiene que
  usar las condiciones.
- Rivales: el mismo HCMG sin condiciones (ceros), y una U-Net de la industria que recibe las condiciones como
  canales extra (lo más simple), con atajos entre capas (HCMG no tiene).
"""

from __future__ import annotations

import math

import numpy as np


# ================================================================ D-ReG
def _modelo_texto(nombre: str, v: int, d: int = 128, ventana: int = 16):
    import torch
    from torch import nn

    if nombre == "gru_grande":  # GRU con tantos parámetros como D-ReG (comparación justa)
        return _modelo_texto("gru", v, d=200)
    if nombre in ("gru", "lstm"):
        class Recurrente(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(v, d)
                self.rnn = (nn.GRU if nombre == "gru" else nn.LSTM)(d, d, batch_first=True)
                self.salida = nn.Linear(d, v)

            def forward(self, x):
                return self.salida(self.rnn(self.emb(x))[0])
        return Recurrente()

    if nombre == "transformer":
        class Causal(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(v, d)
                self.pos = nn.Parameter(torch.randn(1, 64, d) * 0.02)
                capa = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.1, batch_first=True)
                self.cuerpo = nn.TransformerEncoder(capa, 2)
                self.salida = nn.Linear(d, v)

            def forward(self, x):
                n = x.shape[1]
                m = nn.Transformer.generate_square_subsequent_mask(n, device=x.device)
                return self.salida(self.cuerpo(self.emb(x) + self.pos[:, :n], mask=m, is_causal=True))
        return Causal()

    class CIU(nn.Module):  # CrossInteractionUnit del cuaderno
        def __init__(self, dq, dk, dout):
            super().__init__()
            self.atn = nn.MultiheadAttention(dq, 4, batch_first=True)
            self.norma, self.lineal = nn.LayerNorm(dq), nn.Linear(dq, dout)
            self.k, self.v = nn.Linear(dk, dq), nn.Linear(dk, dq)

        def forward(self, q, influencia):
            a, _ = self.atn(q.unsqueeze(1), self.k(influencia).unsqueeze(1), self.v(influencia).unsqueeze(1))
            return torch.relu(self.lineal(self.norma(q + a.squeeze(1))))

    arreglado = nombre == "dreg_arreglado"

    class DReG(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(v, d)
            self.p_atn = nn.MultiheadAttention(d, 4, batch_first=True)
            self.p_ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))
            self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.gru = nn.GRUCell(d, d)
            self.p_a_c, self.c_a_p = CIU(d, d, d), CIU(d, d, d)
            self.salida = nn.Linear(d, v)

        def forward(self, x):
            b, n = x.shape
            e = self.emb(x)
            hp, hc = e.new_zeros(b, d), e.new_zeros(b, d)
            historia, salidas = [], []
            for t in range(n):
                entrada = self.c_a_p(hp, hc) + e[:, t]
                if arreglado:  # atiende a las últimas `ventana` entradas (incluida la actual)
                    historia.append(entrada)
                    memoria = torch.stack(historia[-ventana:], 1)
                    a, _ = self.p_atn(entrada.unsqueeze(1), memoria, memoria)
                else:  # tal cual: atención sobre un solo elemento
                    q = entrada.unsqueeze(1)
                    a, _ = self.p_atn(q, q, q)
                hp = self.n2(self.n1(a.squeeze(1)) + self.p_ffn(self.n1(a.squeeze(1))))
                hc = self.gru(self.p_a_c(hc, hp), hc)
                salidas.append(hp)
            return self.salida(torch.stack(salidas, 1))
    return DReG()


def _dispositivo(dispositivo: str):
    import torch

    if dispositivo == "auto":
        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(dispositivo)


def entrenar_texto(nombre: str, xe, v: int, semilla: int = 0, pasos: int = 1500, lote: int = 32, lr: float = 1e-3,
                   dispositivo: str = "auto"):
    import torch
    from torch import nn

    from .difusion_texto import PAD

    torch.manual_seed(semilla)
    rng = np.random.default_rng(semilla)
    dev = _dispositivo(dispositivo)
    m = _modelo_texto(nombre, v).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for _ in range(pasos):
        x = torch.as_tensor(xe[rng.integers(0, len(xe), lote)], device=dev)
        logits = m(x[:, :-1])
        perdida = nn.functional.cross_entropy(logits.reshape(-1, v), x[:, 1:].reshape(-1), ignore_index=PAD)
        opt.zero_grad()
        perdida.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    return m


def perplejidad_texto(m, xp, v: int) -> float:
    import torch
    from torch import nn

    from .difusion_texto import PAD

    m.eval()
    dev = next(m.parameters()).device
    total, cuantos = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(xp), 128):
            x = torch.as_tensor(xp[i:i + 128], device=dev)
            logits = m(x[:, :-1])
            total += float(nn.functional.cross_entropy(logits.reshape(-1, v), x[:, 1:].reshape(-1),
                                                       ignore_index=PAD, reduction="sum"))
            cuantos += int((x[:, 1:] != PAD).sum())
    return math.exp(total / cuantos)


def competir_dreg(entreno: int = 3000, prueba: int = 500, semillas=(0, 1, 2), pasos: int = 1500, aviso=print,
                  modelos=("dreg_cuaderno", "dreg_arreglado", "gru", "gru_grande", "lstm", "transformer"),
                  dispositivo: str = "auto") -> dict:
    from .datos_reales import textos_reales
    from .difusion_texto import preparar

    te, _, tp, _ = textos_reales("resenas_es", entreno, prueba)
    tok, xe, xp = preparar(te, tp, vocabulario=2000, largo=24)
    v = tok.get_vocab_size()
    r = {"_datos": {"entreno": len(xe), "prueba": len(xp), "vocabulario": v, "pasos": pasos, "semillas": len(semillas)}}
    for nombre in modelos:
        filas = []
        for s in semillas:
            m = entrenar_texto(nombre, xe, v, s, pasos, dispositivo=dispositivo)
            filas.append({"perplejidad": perplejidad_texto(m, xp, v), "parametros": sum(p.numel() for p in m.parameters())})
        r[nombre] = filas
        aviso(f"{nombre:16} perplejidad {np.mean([f['perplejidad'] for f in filas]):.1f}")
    return r


# ================================================================ HCMG
def datos_hcmg(n: int, semilla: int = 0, parte: str = "entreno"):
    """(borroso [n,3,16,16], clase one-hot [n,10], color [n,3], objetivo [n,3,16,16]) a partir de los dígitos.
    ``parte`` elige el grupo de dígitos ("entreno" o "prueba"); los dos grupos no comparten ningún dígito."""
    import torch
    import torch.nn.functional as F
    from sklearn.datasets import load_digits

    from .video import indices_digitos

    d = load_digits()
    rng = np.random.default_rng(semilla)
    grupo = indices_digitos(parte)
    idx = grupo[rng.integers(0, len(grupo), n)]
    img = torch.as_tensor(np.kron(d.images[idx] / 16.0, np.ones((2, 2))), dtype=torch.float32)  # [n,16,16]
    color = torch.as_tensor(rng.uniform(0.2, 1.0, (n, 3)), dtype=torch.float32)
    objetivo = img[:, None] * color[:, :, None, None] * 2 - 1  # en [-1, 1] como la salida tanh
    borroso = F.avg_pool2d(img[:, None], 8)  # 2×2: solo queda una mancha
    borroso = F.interpolate(borroso, size=16, mode="bilinear", align_corners=False).expand(-1, 3, -1, -1) * 2 - 1
    clase = F.one_hot(torch.as_tensor(d.target[idx]), 10).float()
    return borroso.contiguous(), clase, color, objetivo


def _modelo_imagen(nombre: str, base: int = 32):
    import torch
    from torch import nn

    if nombre == "unet_condiciones":
        def bloque(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.ReLU(), nn.Conv2d(b, b, 3, padding=1), nn.ReLU())

        class UNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.e1, self.e2, self.e3 = bloque(3 + 13, base), bloque(base, 2 * base), bloque(2 * base, 4 * base)
                self.s2, self.d2 = nn.ConvTranspose2d(4 * base, 2 * base, 2, 2), bloque(4 * base, 2 * base)
                self.s1, self.d1 = nn.ConvTranspose2d(2 * base, base, 2, 2), bloque(2 * base, base)
                self.salida = nn.Conv2d(base, 3, 1)

            def forward(self, x, clase, color):
                c = torch.cat([clase, color], 1)[:, :, None, None].expand(-1, -1, *x.shape[2:])
                a = self.e1(torch.cat([x, c], 1))
                b = self.e2(nn.functional.max_pool2d(a, 2))
                cc = self.e3(nn.functional.max_pool2d(b, 2))
                d = self.d2(torch.cat([self.s2(cc), b], 1))
                return torch.tanh(self.salida(self.d1(torch.cat([self.s1(d), a], 1))))
        return UNet()

    class NormAdaptativa(nn.Module):  # AdaptiveNormalization del cuaderno
        def __init__(self, canales, dim):
            super().__init__()
            self.norma = nn.InstanceNorm2d(canales, affine=False)
            self.g, self.b = nn.Linear(dim, canales), nn.Linear(dim, canales)

        def forward(self, x, c):
            return self.norma(x) * (1 + self.g(c)[:, :, None, None]) + self.b(c)[:, :, None, None]

    class BFC(nn.Module):  # ConditionalFusionBlock del cuaderno
        def __init__(self, a, b):
            super().__init__()
            self.conv = nn.Conv2d(a, b, 3, padding=1)
            self.n_id, self.n_texto = NormAdaptativa(b, 10), NormAdaptativa(b, 3)

        def forward(self, x, clase, color):
            return torch.relu(self.n_texto(self.n_id(self.conv(x), clase), color))

    sin_condicion = nombre == "hcmg_sin_condicion"

    class HCMG(nn.Module):
        def __init__(self):
            super().__init__()
            self.ini = nn.Conv2d(3, base, 3, padding=1)
            self.b1, self.baja1 = BFC(base, 2 * base), nn.Conv2d(2 * base, 2 * base, 4, 2, 1)
            self.b2, self.baja2 = BFC(2 * base, 4 * base), nn.Conv2d(4 * base, 4 * base, 4, 2, 1)
            self.b3 = BFC(4 * base, 8 * base)
            self.sube1, self.g1 = nn.ConvTranspose2d(8 * base, 4 * base, 4, 2, 1), BFC(4 * base, 2 * base)
            self.sube2, self.g2 = nn.ConvTranspose2d(2 * base, base, 4, 2, 1), BFC(base, base)
            self.fin = nn.Conv2d(base, 3, 3, padding=1)

        def forward(self, x, clase, color):
            if sin_condicion:
                clase, color = torch.zeros_like(clase), torch.zeros_like(color)
            x = self.ini(x)
            x = self.baja1(self.b1(x, clase, color))
            x = self.baja2(self.b2(x, clase, color))
            x = self.b3(x, clase, color)
            x = self.g1(self.sube1(x), clase, color)
            x = self.g2(self.sube2(x), clase, color)
            return torch.tanh(self.fin(x))
    return HCMG()


def competir_hcmg(entreno: int = 3000, prueba: int = 500, semillas=(0, 1, 2), epocas: int = 20, aviso=print,
                  dispositivo: str = "auto") -> dict:
    import torch
    from torch import nn

    from .video import psnr, ssim

    dev = _dispositivo(dispositivo)
    xe, ce, ke, ye = (t.to(dev) for t in datos_hcmg(entreno, 0, "entreno"))
    xp, cp, kp, yp = (t.to(dev) for t in datos_hcmg(prueba, 1, "prueba"))
    r = {"_datos": {"entreno": entreno, "prueba": prueba, "epocas": epocas, "semillas": len(semillas)}}
    for nombre in ("hcmg", "hcmg_sin_condicion", "unet_condiciones"):
        filas = []
        for s in semillas:
            torch.manual_seed(s)
            rng = np.random.default_rng(s)
            m = _modelo_imagen(nombre).to(dev)
            opt = torch.optim.Adam(m.parameters(), lr=1e-3)
            for _ in range(epocas):
                orden = torch.as_tensor(rng.permutation(entreno), device=dev)
                for i in range(0, entreno, 64):
                    j = orden[i:i + 64]
                    perdida = nn.functional.mse_loss(m(xe[j], ce[j], ke[j]), ye[j])
                    opt.zero_grad(); perdida.backward(); opt.step()
            m.eval()
            with torch.no_grad():
                pred = (m(xp, cp, kp) + 1) / 2
            pred, real = pred.cpu(), ((yp + 1) / 2).cpu()
            filas.append({"psnr": psnr(pred.numpy(), real.numpy()), "ssim": ssim(pred, real),
                          "parametros": sum(p.numel() for p in m.parameters())})
        r[nombre] = filas
        aviso(f"{nombre:20} PSNR {np.mean([f['psnr'] for f in filas]):.2f}")
    return r


# ================================================================ informe
def informe(dreg: dict, hcmg: dict) -> str:
    def f(filas, k, dec=1):
        a = np.array([x[k] for x in filas])
        return (f"{a.mean():.{dec}f} ± {a.std():.{dec}f}").replace(".", ",")

    dd, dh = dreg["_datos"], hcmg["_datos"]
    l = ["# Cuaderno 07: D-ReG y HCMG", "",
         f"## D-ReG: predecir la palabra siguiente en reseñas reales ({dd['entreno']} para entrenar, {dd['prueba']} de "
         f"prueba, BPE de {dd['vocabulario']}, {dd['pasos']} pasos, {dd['semillas']} semillas)", "",
         "| Modelo | Perplejidad (menos es mejor) | Parámetros |", "|---|---|---|"]
    nombres = {"dreg_cuaderno": "D-ReG tal cual (atención sobre 1 elemento)", "dreg_arreglado": "D-ReG con atención a las últimas 16 palabras (arreglo)",
               "gru": "GRU (industria clásica)", "gru_grande": "GRU del mismo tamaño que D-ReG (industria)", "lstm": "LSTM (industria clásica)", "transformer": "Transformer causal, como GPT (industria)"}
    for k, n in nombres.items():
        if k not in dreg:
            continue
        l.append(f"| {n} | {f(dreg[k], 'perplejidad')} | {format(dreg[k][0]['parametros'], ',').replace(',', ' ')} |")
    l += ["", f"## HCMG: dibujar el dígito nítido de la clase y el color pedidos a partir de un borrón ({dh['entreno']} "
          f"para entrenar, {dh['prueba']} de prueba, {dh['epocas']} épocas, {dh['semillas']} semillas)", "",
          "| Modelo | PSNR (más es mejor) | SSIM | Parámetros |", "|---|---|---|---|"]
    nombres = {"hcmg": "HCMG tal cual (dos normalizaciones condicionadas)", "hcmg_sin_condicion": "HCMG sin condiciones (control)",
               "unet_condiciones": "U-Net con las condiciones como canales (industria)"}
    for k, n in nombres.items():
        l.append(f"| {n} | {f(hcmg[k], 'psnr', 2)} | {f(hcmg[k], 'ssim', 3)} | {format(hcmg[k][0]['parametros'], ',').replace(',', ' ')} |")
    return "\n".join(l) + "\n"
