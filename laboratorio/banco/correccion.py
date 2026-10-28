"""Corrección ortográfica (cuaderno 01, «Clasificador BERT con corrección GNN»).

Tu idea: cada letra de la frase es un nodo de un grafo en cadena; una
auto-atención y tres capas GCN predicen la letra correcta en cada posición.

Problemas del original:
- **Recompensa con ruido**: sumaba un «puntaje BERT» cuya capa final nunca se entrenó.
- **Una sola frase**: entrenaba 100 000 veces sobre la misma frase.
- **Límite de diseño**: predice una letra por cada letra de entrada, así que solo
  puede arreglar letras *cambiadas*, no letras de más ni de menos. Por eso hay
  dos escenarios.

Compiten, con frases reales de las reseñas y los errores de tu propia función
``introducir_errores_en_frase``:

| Corrector | Qué es |
|---|---|
| ``sin_corregir`` | deja el texto como está (el piso) |
| ``diccionario`` | corrector clásico (Norvig, SymSpell): cambia cada palabra desconocida por la más frecuente a 1–2 ediciones |
| ``gnn_rl`` | tu GNN con auto-atención, entrenada con refuerzo como en el cuaderno (recompensa = parecido Levenshtein, sin BERT) |
| ``gnn`` | tu misma GNN entrenada de forma supervisada |
| ``gnn_residual`` | tu GNN con conexiones residuales (el arreglo: sin ellas la red olvida la letra original) |
| ``bilstm`` / ``transformer`` | etiquetadores de caracteres de la industria (misma tarea, letra por letra) |

Métricas: **CER** (% de letras mal tras corregir; menos es mejor), **palabras
correctas**, **frases perfectas** y velocidad.
"""

from __future__ import annotations

import random
import re
import time
from collections import Counter
from statistics import mean, stdev

import torch
import torch.nn.functional as F
from torch import nn

LETRAS = list("abcdefghijklmnopqrstuvwxyzáéíóúñ ")  # el vocabulario del cuaderno (el espacio es la última)
INDICE = {c: i + 1 for i, c in enumerate(LETRAS)}  # 0 = relleno
INVERSO = {i: c for c, i in INDICE.items()}
LARGO_MAX = 128


# ---------------------------------------------------------------- Datos (función del cuaderno, sin cambios)
def introducir_errores_en_frase(texto, max_errores_por_palabra=1, prob_error_espacio=0.1, rng=random,
                                tipos=("sustituir", "insertar", "eliminar", "transponer")):
    """La función del cuaderno 01. Solo se añadió ``rng`` (para repetir) y ``tipos`` (escenarios)."""
    letras = LETRAS
    frase = []
    for palabra in texto.split(" "):
        if not palabra:
            frase.append("")
            continue
        chars = list(palabra)
        for _ in range(rng.randint(0, max_errores_por_palabra)):
            tipo = rng.choice(list(tipos))
            if tipo == "sustituir" and chars:
                chars[rng.randint(0, len(chars) - 1)] = rng.choice(letras[:-1])
            elif tipo == "insertar":
                chars.insert(rng.randint(0, len(chars)), rng.choice(letras[:-1]))
            elif tipo == "eliminar" and chars:
                chars.pop(rng.randint(0, len(chars) - 1))
            elif tipo == "transponer" and len(chars) >= 2:
                i = rng.randint(0, len(chars) - 2)
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
        frase.append("".join(chars))
    final = []
    for i, w in enumerate(frase):
        final.append(w)
        if i < len(frase) - 1:
            if "sustituir" in tipos and len(tipos) == 1:
                final.append(" ")  # escenario «solo letras cambiadas»: los espacios no se tocan
            else:
                final.append(rng.choice(["  ", " ", ""]) if rng.random() < prob_error_espacio else " ")
    return "".join(final).strip()


def limpiar(texto: str) -> str:
    t = "".join(c for c in texto.lower() if c in INDICE)
    return re.sub(" +", " ", t).strip()


def frases_de(textos: list[str], minimo: int = 20, maximo: int = 120) -> list[str]:
    """Frases limpias (solo letras del vocabulario y espacios) de entre 20 y 120 letras."""
    salida = []
    for t in textos:
        for trozo in re.split(r"[.!?\n]+", t):
            f = limpiar(trozo)
            if minimo <= len(f) <= maximo:
                salida.append(f)
    return salida


def preparar(frases: list[str], escenario: str, semilla: int = 1234) -> list[tuple[str, str]]:
    """Pares (con errores, limpia). ``escenario``: "sustituciones" o "todos"."""
    rng = random.Random(semilla)
    tipos = ("sustituir",) if escenario == "sustituciones" else ("sustituir", "insertar", "eliminar", "transponer")
    return [(introducir_errores_en_frase(f, rng.randint(1, 2), 0.2, rng, tipos), f) for f in frases]


# ---------------------------------------------------------------- Métricas
def cer(predicha: str, limpia: str) -> float:
    from rapidfuzz.distance import Levenshtein

    return Levenshtein.distance(predicha, limpia) / max(len(limpia), 1)


def medir(predichas: list[str], pares: list[tuple[str, str]]) -> dict:
    limpias = [l for _, l in pares]
    errores = [cer(p, l) for p, l in zip(predichas, limpias)]
    palabras_ok = palabras_total = 0
    for p, l in zip(predichas, limpias):
        wp, wl = p.split(), l.split()
        palabras_total += len(wl)
        palabras_ok += sum(a == b for a, b in zip(wp, wl)) if len(wp) == len(wl) else 0
    return {"cer": mean(errores), "palabras_ok": palabras_ok / max(palabras_total, 1),
            "perfectas": mean(p == l for p, l in zip(predichas, limpias))}


# ---------------------------------------------------------------- Corrector por diccionario (industria clásica)
class CorrectorDiccionario:
    """Al estilo de Peter Norvig / SymSpell: palabra desconocida → la más frecuente a distancia 1, si no, a 2."""

    def __init__(self, frases: list[str]) -> None:
        self.frec = Counter(w for f in frases for w in f.split())
        self.letras = "".join(LETRAS[:-1])

    def _ediciones1(self, w: str) -> set[str]:
        cortes = [(w[:i], w[i:]) for i in range(len(w) + 1)]
        borrar = [a + b[1:] for a, b in cortes if b]
        trasponer = [a + b[1] + b[0] + b[2:] for a, b in cortes if len(b) > 1]
        cambiar = [a + c + b[1:] for a, b in cortes if b for c in self.letras]
        insertar = [a + c + b for a, b in cortes for c in self.letras]
        return set(borrar + trasponer + cambiar + insertar)

    def palabra(self, w: str) -> str:
        if w in self.frec or not w:
            return w
        e1 = [x for x in self._ediciones1(w) if x in self.frec]
        if e1:
            return max(e1, key=self.frec.__getitem__)
        if len(w) <= 12:
            e2 = {y for x in self._ediciones1(w) for y in self._ediciones1(x) if y in self.frec}
            if e2:
                return max(e2, key=self.frec.__getitem__)
        return w

    def corregir(self, frase: str) -> str:
        return " ".join(self.palabra(w) for w in frase.split())


# ---------------------------------------------------------------- Redes (letra por letra)
def _codificar(frases: list[str]) -> torch.Tensor:
    n = min(max(len(f) for f in frases), LARGO_MAX)
    m = torch.zeros(len(frases), n, dtype=torch.long)
    for i, f in enumerate(frases):
        for j, c in enumerate(f[:n]):
            m[i, j] = INDICE.get(c, INDICE[" "])
    return m


class GCNCadena(nn.Module):
    """GCNConv (Kipf y Welling, 2017) sobre un grafo en cadena: cada letra conectada con la anterior y la
    siguiente, con lazo propio. Salida = D^-1/2 (A + I) D^-1/2 · X · W + b, igual que PyTorch Geometric."""

    def __init__(self, entradas: int, salidas: int) -> None:
        super().__init__()
        self.lin = nn.Linear(entradas, salidas, bias=False)
        self.bias = nn.Parameter(torch.zeros(salidas))

    def forward(self, x, mascara):  # x: [B, L, D]; mascara: [B, L] (True = letra real)
        h = self.lin(x) * mascara.unsqueeze(-1)
        m = mascara.float()
        izq = F.pad(m[:, :-1], (1, 0))  # ¿hay vecino a la izquierda?
        der = F.pad(m[:, 1:], (0, 1))
        grado = (1 + izq + der) * m
        inv = grado.clamp(min=1).rsqrt() * m
        hs = h * inv.unsqueeze(-1)
        vecinos = hs + F.pad(hs[:, :-1], (0, 0, 1, 0)) + F.pad(hs[:, 1:], (0, 0, 0, 1))
        return vecinos * inv.unsqueeze(-1) + self.bias


class CorreccionGNN(nn.Module):
    """Tu red: embeddings 16 → auto-atención de 4 cabezas → 3 GCN de 32 → letra. (La auto-atención del cuaderno
    trabajaba frase por frase; aquí es la de PyTorch con máscara, que hace lo mismo para lotes.)"""

    def __init__(self, emb: int = 16, oculto: int = 32, capas: int = 3, cabezas: int = 4) -> None:
        super().__init__()
        self.emb = nn.Embedding(len(LETRAS) + 1, emb, padding_idx=0)
        self.atencion = nn.MultiheadAttention(emb, cabezas, batch_first=True)
        self.convs = nn.ModuleList([GCNCadena(emb, oculto)] + [GCNCadena(oculto, oculto) for _ in range(capas - 1)])
        self.decoder = nn.Linear(oculto, len(LETRAS) + 1)

    def forward(self, ids):
        mascara = ids != 0
        x = self.emb(ids)
        x, _ = self.atencion(x, x, x, key_padding_mask=~mascara)
        for i, conv in enumerate(self.convs):
            x = conv(x, mascara)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return self.decoder(x)


class CorreccionGNNResidual(nn.Module):
    """Tu misma red con conexiones residuales (el cuaderno las dejó comentadas como «opcional»).

    Sin ellas, la salida de la atención reemplaza a la letra original y las 3 GCN promedian tanto a los vecinos
    que la red olvida qué letra había: corrige peor que no corregir. Con ellas, cada capa *suma* su arreglo.
    """

    def __init__(self, emb: int = 16, oculto: int = 32, capas: int = 3, cabezas: int = 4) -> None:
        super().__init__()
        self.emb = nn.Embedding(len(LETRAS) + 1, emb, padding_idx=0)
        self.atencion = nn.MultiheadAttention(emb, cabezas, batch_first=True)
        self.proy = nn.Linear(emb, oculto)
        self.convs = nn.ModuleList(GCNCadena(oculto, oculto) for _ in range(capas))
        self.decoder = nn.Linear(oculto, len(LETRAS) + 1)

    def forward(self, ids):
        mascara = ids != 0
        e = self.emb(ids)
        a, _ = self.atencion(e, e, e, key_padding_mask=~mascara)
        h = self.proy(e + a)
        for conv in self.convs:
            h = h + F.relu(conv(h, mascara))
        return self.decoder(h)


class EtiquetadorBiLSTM(nn.Module):
    def __init__(self, emb: int = 32, oculto: int = 64) -> None:
        super().__init__()
        self.emb = nn.Embedding(len(LETRAS) + 1, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, oculto, batch_first=True, bidirectional=True, num_layers=2)
        self.salida = nn.Linear(2 * oculto, len(LETRAS) + 1)

    def forward(self, ids):
        return self.salida(self.lstm(self.emb(ids))[0])


class EtiquetadorTransformer(nn.Module):
    def __init__(self, dim: int = 64, capas: int = 3) -> None:
        super().__init__()
        self.emb = nn.Embedding(len(LETRAS) + 1, dim, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, LARGO_MAX, dim))
        capa = nn.TransformerEncoderLayer(dim, 4, 4 * dim, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(capa, capas, enable_nested_tensor=False)
        self.salida = nn.Linear(dim, len(LETRAS) + 1)

    def forward(self, ids):
        return self.salida(self.enc(self.emb(ids) + self.pos[:, : ids.size(1)], src_key_padding_mask=ids == 0))


REDES = {"gnn": CorreccionGNN, "gnn_rl": CorreccionGNN, "gnn_residual": CorreccionGNNResidual,
         "bilstm": EtiquetadorBiLSTM, "transformer": EtiquetadorTransformer}


def _decodificar(logits: torch.Tensor, ids: torch.Tensor) -> list[str]:
    pred = logits.argmax(-1).cpu()
    salida = []
    for fila, orig in zip(pred, ids.cpu()):
        n = int((orig != 0).sum())
        salida.append(re.sub(" +", " ", "".join(INVERSO.get(int(i), " ") for i in fila[:n])).strip())
    return salida


def predecir(modelo: nn.Module, frases: list[str], lote: int = 256) -> list[str]:
    modelo.eval()
    disp = next(modelo.parameters()).device
    salida = []
    with torch.no_grad():
        for i in range(0, len(frases), lote):
            ids = _codificar(frases[i : i + lote]).to(disp)
            salida += _decodificar(modelo(ids), ids)
    return salida


def _objetivo_alineado(pares: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
    """Para entrenar letra por letra hacen falta frases del mismo largo (solo las sustituciones lo cumplen)."""
    sucias, limpias = zip(*[(s, l) for s, l in pares if len(s) == len(l)]) if pares else ((), ())
    return list(sucias), list(limpias)


def entrenar(nombre: str, pares: list[tuple[str, str]], semilla: int, epocas: int = 30, lote: int = 64,
             paciencia: int = 4, muestras_rl: int = 8, dispositivo: str = "auto") -> tuple[nn.Module, dict]:
    """Entrena con 90 % de los pares y usa 10 % para parar a tiempo (CER de validación)."""
    from rapidfuzz.distance import Levenshtein

    disp = torch.device("cuda" if dispositivo == "auto" and torch.cuda.is_available() else
                        ("cpu" if dispositivo == "auto" else dispositivo))
    torch.manual_seed(semilla)
    gen = torch.Generator().manual_seed(semilla)
    n_val = max(1, len(pares) // 10)
    val, ent = pares[:n_val], pares[n_val:]
    sucias, limpias = _objetivo_alineado(ent)
    if not sucias:
        raise ValueError("No hay pares del mismo largo para entrenar letra por letra")
    x, y = _codificar(sucias), _codificar(limpias)
    modelo = REDES[nombre]().to(disp)
    opt = torch.optim.Adam(modelo.parameters(), lr=0.002 if nombre != "gnn_rl" else 0.0005)
    mejor, estado, sin_mejora, epoca = float("inf"), None, 0, 0
    inicio = time.perf_counter()
    for epoca in range(epocas):
        modelo.train()
        for idx in torch.randperm(len(x), generator=gen).split(lote):
            xb, yb = x[idx].to(disp), y[idx].to(disp)
            logits = modelo(xb)
            if nombre == "gnn_rl":
                # Como el cuaderno: varias frases muestreadas, recompensa = 1 − Levenshtein normalizado,
                # REINFORCE con la mejor contra el promedio. (Sin «puntaje BERT»: era una capa sin entrenar.)
                dist = torch.distributions.Categorical(logits=logits)
                validas = xb != 0
                objetivos = [limpias[i] for i in idx.tolist()]
                muestras, logps, recompensas = [], [], []
                for _ in range(muestras_rl):
                    m = dist.sample()
                    logps.append((dist.log_prob(m) * validas).sum(1) / validas.sum(1))
                    textos = _decodificar(F.one_hot(m, len(LETRAS) + 1).float(), xb)
                    recompensas.append(torch.tensor([1 - Levenshtein.normalized_distance(t, o)
                                                     for t, o in zip(textos, objetivos)], device=disp))
                r = torch.stack(recompensas)  # [muestras, B]
                lp = torch.stack(logps)
                mejor_i = r.argmax(0)
                ventaja = r.gather(0, mejor_i[None]).squeeze(0) - r.mean(0)
                loss = -(ventaja * lp.gather(0, mejor_i[None]).squeeze(0)).mean()
            else:
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1), ignore_index=0)
            opt.zero_grad()
            loss.backward()
            opt.step()
        cer_val = medir(predecir(modelo, [s for s, _ in val]), val)["cer"]
        if cer_val < mejor:
            mejor, sin_mejora = cer_val, 0
            estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    modelo.load_state_dict(estado)
    return modelo, {"segundos_entrenar": round(time.perf_counter() - inicio, 1), "epocas": epoca + 1,
                    "parametros": sum(p.numel() for p in modelo.parameters()), "cer_validacion": round(mejor, 4)}


# ---------------------------------------------------------------- Competencia
def competir_correctores(frases_e: list[str], frases_p: list[str], escenarios=("sustituciones", "todos"),
                         redes=("gnn", "gnn_rl", "gnn_residual", "bilstm", "transformer"), semillas=(0, 1, 2), aviso=print,
                         **entreno) -> dict:
    dicc = CorrectorDiccionario(frases_e)
    resultado = {"escenarios": {}, "frases_entreno": len(frases_e), "frases_prueba": len(frases_p),
                 "semillas": len(semillas)}
    for esc in escenarios:
        pares_e, pares_p = preparar(frases_e, esc, 1234), preparar(frases_p, esc, 4321)
        sucias = [s for s, _ in pares_p]
        filas = []
        for nombre, fn in (("sin_corregir", lambda f: f), ("diccionario", dicc.corregir)):
            t0 = time.perf_counter()
            pred = [fn(s) for s in sucias]
            seg = time.perf_counter() - t0
            m = medir(pred, pares_p)
            filas.append({"nombre": nombre, **{k: [v] for k, v in m.items()}, "frases_por_s": len(sucias) / seg})
            aviso(f"{esc:13} {nombre:13} CER={m['cer']:.4f}")
        for red in redes:
            acc: dict = {"cer": [], "palabras_ok": [], "perfectas": []}
            info, vel = [], []
            for s in semillas:
                modelo, datos = entrenar(red, pares_e, s, **entreno)
                t0 = time.perf_counter()
                pred = predecir(modelo, sucias)
                vel.append(len(sucias) / (time.perf_counter() - t0))
                for k, v in medir(pred, pares_p).items():
                    acc[k].append(v)
                info.append(datos)
            filas.append({"nombre": red, **acc, "frases_por_s": mean(vel), "parametros": info[0]["parametros"],
                          "segundos_entrenar": mean(d["segundos_entrenar"] for d in info)})
            aviso(f"{esc:13} {red:13} CER={mean(acc['cer']):.4f}")
        resultado["escenarios"][esc] = filas
    return resultado


def informe_correctores(r: dict) -> str:
    def fmt(v):
        return f"{mean(v) * 100:.2f} %" + (f" ± {stdev(v) * 100:.2f}" if len(v) > 1 else "")

    nombres = {"sin_corregir": "Sin corregir (piso)", "diccionario": "Diccionario (Norvig/SymSpell)",
               "gnn": "Tu GNN tal cual, supervisada", "gnn_rl": "Tu GNN tal cual, con refuerzo (como el cuaderno)",
               "gnn_residual": "Tu GNN con conexiones residuales (arreglo)",
               "bilstm": "BiLSTM de caracteres", "transformer": "Transformer de caracteres"}
    lineas = ["# Corrección ortográfica: tu GNN contra la industria", "",
              f"Frases reales de reseñas ({r['frases_entreno']:,} para entrenar, {r['frases_prueba']:,} de prueba) "
              f"con los errores de tu función `introducir_errores_en_frase`; {r['semillas']} semillas. "
              "**CER** = % de letras mal después de corregir (menos es mejor).", ""]
    titulos = {"sustituciones": "Escenario 1: solo letras cambiadas (justo para corregir letra por letra)",
               "todos": "Escenario 2: todos los errores (letras cambiadas, de más, de menos, traspuestas y espacios)"}
    for esc, filas in r["escenarios"].items():
        lineas += [f"## {titulos.get(esc, esc)}", "",
                   "| Corrector | CER | Palabras correctas | Frases perfectas | Frases/s | Parámetros | Seg. entrenar |",
                   "|---|---|---|---|---|---|---|"]
        for f in sorted(filas, key=lambda f: mean(f["cer"])):
            extra = (f"{f['parametros']:,} | {f['segundos_entrenar']:.0f}" if "parametros" in f else "— | —")
            lineas.append(f"| {nombres.get(f['nombre'], f['nombre'])} | {fmt(f['cer'])} | {fmt(f['palabras_ok'])} | "
                          f"{fmt(f['perfectas'])} | {f['frases_por_s']:,.0f} | {extra} |")
        lineas.append("")
    lineas.append("Las redes de este banco corrigen letra por letra: no pueden quitar ni añadir letras, así que en el "
                  "escenario 2 tienen un techo. El diccionario corrige palabra por palabra y sí puede.")
    return "\n".join(lineas) + "\n"
