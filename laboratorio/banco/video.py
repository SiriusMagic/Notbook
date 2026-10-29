"""Cuaderno 04 · Interpolación y predicción de fotogramas de video.

El cuaderno tiene tres modelos para «inventar» un fotograma:

- **Tripletas** (``ImageReconstructionModel``): una CNN de 3 capas ve el fotograma anterior y el siguiente y dibuja
  el del medio.
- **FrameInterpolator** (celda 13 y celda 446 del cuaderno completo): ve 4 fotogramas (i, i+1, i+3, i+4), los
  aplana en un vector y los pasa por una LSTM de 2 capas (1 024 neuronas) con secuencias de largo 1, y una capa
  lineal dibuja el fotograma i+2. Con secuencias de largo 1 la LSTM no recuerda nada: es una red densa enorme que
  no sabe que los píxeles vecinos están cerca.
- **FrameInterpolator de una entrada** (celdas 42-43): mismo modelo, pero ve 1 fotograma y dibuja el siguiente.

Datos: videos de dígitos que se mueven y rebotan (como *Moving MNIST*, la prueba clásica de video), hechos con los
dígitos de scikit-learn, 32×32 a color, 5 fotogramas por clip. Sin descargas. Los dígitos de los clips de prueba
salen de un grupo distinto al de entrenamiento (nunca se vieron al entrenar).

Rivales:
- **copiar**: repetir el fotograma más cercano; **promedio**: mezclar el anterior y el siguiente (sin red).
- **U-Net** (industria): convoluciones con atajos entre capas (la base de las redes de interpolación actuales, como
  RIFE o FILM), que aprende la **corrección** sobre el promedio de los vecinos.
- **tripletas_residual** (arreglo): la misma CNN de tripletas, pero aprende la corrección sobre el promedio.

Métricas en clips de prueba: PSNR (en dB; más es mejor; +3 dB = la mitad de error) y SSIM (parecido estructural;
1 = idéntico).
"""

from __future__ import annotations

import math

import numpy as np

LADO = 32


# ---------------------------------------------------------------- datos
def indices_digitos(parte: str) -> np.ndarray:
    """Reparte los 1 797 dígitos de scikit-learn en dos grupos que no se tocan: 70 % para entrenar y 30 % para la
    prueba. Así los dígitos de los clips de prueba nunca se vieron al entrenar."""
    orden = np.random.default_rng(1234).permutation(1797)
    corte = int(0.7 * len(orden))
    return orden[:corte] if parte == "entreno" else orden[corte:]


def clips(n: int, semilla: int = 0, fotogramas: int = 5, digitos: int = 2, parte: str = "entreno") -> np.ndarray:
    """``n`` clips [n, fotogramas, 3, 32, 32] con ``digitos`` dígitos de colores que se mueven y rebotan, tomados
    solo del grupo ``parte`` ("entreno" o "prueba")."""
    from sklearn.datasets import load_digits

    imgs = load_digits().images[indices_digitos(parte)] / 16.0  # 8×8
    imgs = np.kron(imgs, np.ones((2, 2)))  # 16×16
    rng = np.random.default_rng(semilla)
    out = np.zeros((n, fotogramas, 3, LADO, LADO), dtype=np.float32)
    lado = imgs.shape[1]
    tope = LADO - lado
    for k in range(n):
        for _ in range(digitos):
            sprite = imgs[rng.integers(len(imgs))]
            color = rng.uniform(0.4, 1.0, 3)
            pos = rng.uniform(0, tope, 2)
            vel = rng.uniform(-3, 3, 2)
            for f in range(fotogramas):
                y, x = int(round(pos[0])), int(round(pos[1]))
                parche = sprite[None] * color[:, None, None]
                zona = out[k, f, :, y:y + lado, x:x + lado]
                np.maximum(zona, parche, out=zona)
                pos += vel
                for d in range(2):  # rebote en los bordes
                    if pos[d] < 0 or pos[d] > tope:
                        vel[d] = -vel[d]
                        pos[d] = np.clip(pos[d], 0, tope)
    return out


# ---------------------------------------------------------------- métricas
def psnr(pred, real) -> float:
    mse = float(((pred - real) ** 2).mean())
    return 100.0 if mse == 0 else 10 * math.log10(1.0 / mse)


def ssim(pred, real) -> float:
    """SSIM con ventana gaussiana 7×7 (como en ``skimage``), promediado sobre canales e imágenes."""
    import torch
    import torch.nn.functional as F

    pred, real = torch.as_tensor(pred, dtype=torch.float32), torch.as_tensor(real, dtype=torch.float32)
    g = torch.exp(-((torch.arange(7) - 3.0) ** 2) / (2 * 1.5**2))
    g = (g[:, None] * g[None]) / g.sum() ** 2
    c = pred.shape[1]
    w = g.expand(c, 1, 7, 7)
    media = lambda x: F.conv2d(x, w, groups=c)  # noqa: E731
    mx, my = media(pred), media(real)
    vx, vy = media(pred * pred) - mx**2, media(real * real) - my**2
    cxy = media(pred * real) - mx * my
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx**2 + my**2 + c1) * (vx + vy + c2))
    return float(s.mean())


# ---------------------------------------------------------------- modelos
def _modelo(nombre: str, entradas: int, ocultas_lstm: int = 1024):
    import torch
    from torch import nn

    if nombre in ("tripletas", "tripletas_residual"):
        class Tripletas(nn.Module):  # ImageReconstructionModel del cuaderno
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(nn.Conv2d(3 * entradas, 32, 3, padding=1), nn.ReLU(),
                                         nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 3, 1))
                self.residual = nombre == "tripletas_residual"

            def forward(self, x):
                y = self.net(x)
                return y + _promedio(x) if self.residual else y
        return Tripletas()

    if nombre == "lstm_cuaderno":
        class FrameInterpolator(nn.Module):  # celda 13 / 42 del cuaderno
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(3 * entradas * LADO * LADO, ocultas_lstm, num_layers=2, batch_first=True)
                self.fc = nn.Linear(ocultas_lstm, 3 * LADO * LADO)

            def forward(self, x):
                _, (h, _) = self.lstm(x.flatten(1).unsqueeze(1))
                return self.fc(h[-1]).view(-1, 3, LADO, LADO)
        return FrameInterpolator()

    if nombre == "tripletas_dilatadas":
        class TripletasDilatadas(nn.Module):
            """Mejora 1: la misma CNN chica, pero con convoluciones dilatadas (1, 2, 4, 8): ve hasta ~31 píxeles
            alrededor con casi los mismos parámetros, así alcanza a seguir dígitos que se movieron lejos. Aprende la
            corrección sobre el promedio de los vecinos."""

            def __init__(self, ancho: int = 24):
                super().__init__()
                capas, c = [], 3 * entradas
                for d in (1, 2, 4, 8):
                    capas += [nn.Conv2d(c, ancho, 3, padding=d, dilation=d), nn.ReLU()]
                    c = ancho
                self.net = nn.Sequential(*capas, nn.Conv2d(ancho, 3, 1))

            def forward(self, x):
                return self.net(x) + _promedio(x)
        return TripletasDilatadas()

    if nombre == "tripletas_flujo":
        class TripletasFlujo(nn.Module):
            """Mejora 2: en vez de dibujar el fotograma, estima hacia dónde se movió cada píxel (un «flujo» para el
            fotograma anterior y otro para el siguiente), trae esos píxeles a su lugar (``grid_sample``) y los mezcla
            con una máscara aprendida; una corrección pequeña arregla lo que falte. Es la idea de RIFE o Super SloMo,
            en miniatura. Convoluciones dilatadas para ver movimientos grandes."""

            def __init__(self, ancho: int = 24):
                super().__init__()
                capas, c = [], 3 * entradas
                for d in (1, 2, 4, 8, 1):
                    capas += [nn.Conv2d(c, ancho, 3, padding=d, dilation=d), nn.ReLU()]
                    c = ancho
                self.cuerpo = nn.Sequential(*capas)
                self.flujo = nn.Conv2d(ancho, 2 * entradas, 3, padding=1)   # (dx, dy) por fotograma de entrada
                self.mezcla = nn.Conv2d(ancho, entradas, 3, padding=1)      # cuánto pesa cada fotograma
                self.correccion = nn.Conv2d(ancho, 3, 3, padding=1)
                nn.init.zeros_(self.flujo.weight); nn.init.zeros_(self.flujo.bias)
                nn.init.zeros_(self.correccion.weight); nn.init.zeros_(self.correccion.bias)

            def forward(self, x):
                b, _, h, w = x.shape
                fotos = x.view(b, entradas, 3, h, w)
                z = self.cuerpo(x)
                flujo = torch.tanh(self.flujo(z)).view(b, entradas, 2, h, w) * 0.5  # hasta medio cuadro (en [-1, 1])
                ys, xs = torch.meshgrid(torch.linspace(-1, 1, h, device=x.device),
                                        torch.linspace(-1, 1, w, device=x.device), indexing="ij")
                base = torch.stack([xs, ys], -1)  # [h, w, 2]
                movidas = []
                for i in range(entradas):
                    rejilla = base + flujo[:, i].permute(0, 2, 3, 1)
                    movidas.append(nn.functional.grid_sample(fotos[:, i], rejilla, align_corners=True,
                                                             padding_mode="border"))
                pesos = torch.softmax(self.mezcla(z), 1).unsqueeze(2)  # [b, entradas, 1, h, w]
                return (torch.stack(movidas, 1) * pesos).sum(1) + self.correccion(z)
        return TripletasFlujo()

    if nombre == "unet":
        def bloque(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.ReLU(), nn.Conv2d(b, b, 3, padding=1), nn.ReLU())

        class UNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.e1, self.e2, self.e3 = bloque(3 * entradas, 32), bloque(32, 64), bloque(64, 128)
                self.sube2, self.d2 = nn.ConvTranspose2d(128, 64, 2, 2), bloque(128, 64)
                self.sube1, self.d1 = nn.ConvTranspose2d(64, 32, 2, 2), bloque(64, 32)
                self.salida = nn.Conv2d(32, 3, 1)

            def forward(self, x):
                a = self.e1(x)
                b = self.e2(nn.functional.max_pool2d(a, 2))
                c = self.e3(nn.functional.max_pool2d(b, 2))
                d = self.d2(torch.cat([self.sube2(c), b], 1))
                e = self.d1(torch.cat([self.sube1(d), a], 1))
                return self.salida(e) + _promedio(x)
        return UNet()
    raise ValueError(nombre)


def _promedio(x):
    """Promedio de los fotogramas de entrada (cada uno son 3 canales)."""
    return x.view(x.shape[0], -1, 3, *x.shape[2:]).mean(1)


# ---------------------------------------------------------------- tareas
TAREAS_VIDEO = {
    # nombre: (índices de entrada, índice a dibujar)
    "medio_de_2": ((1, 3), 2),          # tripletas: anterior y siguiente → el del medio
    "medio_de_4": ((0, 1, 3, 4), 2),     # FrameInterpolator: i, i+1, i+3, i+4 → i+2
    "siguiente": ((3,), 4),              # celdas 42-43: un fotograma → el siguiente
}


def _xy(datos, tarea):
    entradas, objetivo = TAREAS_VIDEO[tarea]
    x = datos[:, list(entradas)].reshape(len(datos), -1, LADO, LADO)
    return x, datos[:, objetivo]


def sin_red(nombre: str, datos, tarea: str):
    entradas, objetivo = TAREAS_VIDEO[tarea]
    if nombre == "copiar":
        mas_cerca = min(entradas, key=lambda i: (abs(i - objetivo), -i))
        return datos[:, mas_cerca]
    return datos[:, list(entradas)].mean(1)  # promedio


def entrenar(nombre: str, xe, ye, semilla: int = 0, epocas: int = 30, lote: int = 32, lr: float = 1e-3,
             dispositivo: str = "auto", ocultas_lstm: int = 1024):
    import torch
    from torch import nn

    torch.manual_seed(semilla)
    rng = np.random.default_rng(semilla)
    if dispositivo == "auto":
        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(dispositivo)
    m = _modelo(nombre, xe.shape[1] // 3, ocultas_lstm).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    xe_t, ye_t = torch.as_tensor(xe), torch.as_tensor(ye)
    for _ in range(epocas):
        orden = rng.permutation(len(xe))
        for i in range(0, len(orden), lote):
            j = orden[i:i + lote]
            perdida = nn.functional.mse_loss(m(xe_t[j].to(dev)), ye_t[j].to(dev))
            opt.zero_grad()
            perdida.backward()
            opt.step()
    return m


def predecir(m, x, lote: int = 256):
    import torch

    dev = next(m.parameters()).device
    m.eval()
    with torch.no_grad():
        return np.concatenate([m(torch.as_tensor(x[i:i + lote]).to(dev)).clamp(0, 1).cpu().numpy()
                               for i in range(0, len(x), lote)])


MODELOS = {
    "medio_de_2": ["copiar", "promedio", "tripletas", "tripletas_residual", "tripletas_dilatadas", "tripletas_flujo", "unet"],
    "medio_de_4": ["copiar", "promedio", "lstm_cuaderno", "tripletas_flujo", "unet"],
    "siguiente": ["copiar", "lstm_cuaderno", "unet"],
}


def competir(entreno: int = 2000, prueba: int = 300, semillas=(0, 1, 2), epocas: int = 30, dispositivo: str = "auto",
             ocultas_lstm: int = 1024, aviso=print) -> dict:
    de, dp = clips(entreno, 0, parte="entreno"), clips(prueba, 1, parte="prueba")
    r = {"_datos": {"entreno": entreno, "prueba": prueba, "epocas": epocas, "ocultas_lstm": ocultas_lstm,
                    "semillas": len(semillas)}}
    for tarea, modelos in MODELOS.items():
        xe, ye = _xy(de, tarea)
        xp, yp = _xy(dp, tarea)
        r[tarea] = {}
        for nombre in modelos:
            filas = []
            if nombre in ("copiar", "promedio"):
                pred = sin_red(nombre, dp, tarea)
                filas.append({"psnr": psnr(pred, yp), "ssim": ssim(pred, yp), "parametros": 0})
            else:
                for s in semillas:
                    m = entrenar(nombre, xe, ye, s, epocas, dispositivo=dispositivo, ocultas_lstm=ocultas_lstm)
                    pred = predecir(m, xp)
                    filas.append({"psnr": psnr(pred, yp), "ssim": ssim(pred, yp),
                                  "parametros": sum(p.numel() for p in m.parameters())})
            r[tarea][nombre] = filas
            aviso(f"{tarea:11} {nombre:20} PSNR {np.mean([f['psnr'] for f in filas]):.2f}")
    return r


NOMBRES = {"copiar": "copiar el fotograma más cercano (sin red)", "promedio": "promedio de los vecinos (sin red)",
           "tripletas": "tripletas tal cual (CNN de 3 capas)", "tripletas_residual": "tripletas con corrección sobre el promedio (arreglo)",
           "tripletas_dilatadas": "tripletas dilatadas (mejora 1: ven más lejos)",
           "tripletas_flujo": "tripletas con flujo (mejora 2: mueven los píxeles, como RIFE)",
           "lstm_cuaderno": "FrameInterpolator tal cual (LSTM con la imagen aplanada)", "unet": "U-Net (industria)"}
TITULOS = {"medio_de_2": "Dibujar el fotograma del medio a partir del anterior y el siguiente (tripletas)",
           "medio_de_4": "Dibujar el fotograma del medio a partir de 4 (FrameInterpolator, celda 13)",
           "siguiente": "Predecir el fotograma siguiente a partir de 1 (celdas 42-43)"}


def informe(r: dict) -> str:
    d = r["_datos"]
    lineas = ["# Cuaderno 04: interpolación y predicción de fotogramas", "",
              f"Videos de 2 dígitos que se mueven y rebotan (como Moving MNIST), 32×32 a color: {d['entreno']} clips "
              f"para entrenar y {d['prueba']} de prueba, {d['epocas']} épocas, {d.get('semillas', 3)} semillas. PSNR en dB (más es mejor; "
              "+3 dB = la mitad de error) y SSIM (1 = idéntico).", ""]
    for tarea, titulo in TITULOS.items():
        lineas += [f"## {titulo}", "", "| Modelo | PSNR | SSIM | Parámetros |", "|---|---|---|---|"]
        for nombre, filas in r[tarea].items():
            p = np.array([f["psnr"] for f in filas])
            s = np.array([f["ssim"] for f in filas])
            pm = f"{p.mean():.2f}" + (f" ± {p.std():.2f}" if len(p) > 1 else "")
            sm = f"{s.mean():.3f}" + (f" ± {s.std():.3f}" if len(s) > 1 else "")
            lineas.append(f"| {NOMBRES[nombre]} | {pm} | {sm} | {filas[0]['parametros']:,} |".replace(",", " ")
                          .replace(".", ","))
        lineas.append("")
    return "\n".join(lineas)
