"""Los cinco experimentos del cuaderno 15, las semillas, las estadísticas y el informe.

E1 crecimiento necesario (xor, círculos, lunas) · E2 complejidad localizada (caótica 20D) · E3 cambios de distribución
controlados (lunas rotadas) · E4 aprendizaje continuo (dígitos permutados y partidos; Split/Permuted MNIST en el cuaderno)
· E5 generalización (Fashion-MNIST).

Reglas: datos de prueba que ninguna decisión usa, varias semillas, intervalos de confianza de la diferencia contra cada
comparador y corrección de Holm por comparaciones múltiples. Los umbrales de ``criterios`` se fijaron antes de correr.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .controlador import ABLACIONES, COMPARADORES, Config, Entrenador, Resultado, entrenar

ESTUDIOS = {
    "E1": ["xor", "circulos", "lunas"],
    "E2": ["caotica_20d"],
    "E5": ["fashion_mnist"],
}

# --- aprendizaje continuo y cambios de distribución --------------------------------------------------------------


@dataclass
class Subtarea:
    x_entreno: torch.Tensor
    y_entreno: torch.Tensor
    x_prueba: torch.Tensor
    y_prueba: torch.Tensor
    clases: list | None = None


@dataclass
class Secuencia:
    nombre: str
    tareas: list
    entradas: int
    salidas: int


def _digitos(semilla: int = 1234):
    from sklearn.datasets import load_digits

    d = load_digits()
    x = (d.data / 16.0).astype(np.float32)
    idx = np.random.default_rng(semilla).permutation(len(x))
    corte = int(len(x) * 0.7)
    a, b = idx[:corte], idx[corte:]
    return x[a], d.target[a], x[b], d.target[b]


def digitos_permutados(n_tareas: int = 4, semilla: int = 1234) -> Secuencia:
    """Cada tarea mezcla los píxeles con otra permutación fija: mismas 10 clases, entrada distinta."""
    xe, ye, xp, yp = _digitos(semilla)
    rng = np.random.default_rng(semilla)
    tareas = []
    for t in range(n_tareas):
        p = np.arange(64) if t == 0 else rng.permutation(64)
        tareas.append(Subtarea(torch.tensor(xe[:, p]), torch.tensor(ye), torch.tensor(xp[:, p]), torch.tensor(yp)))
    return Secuencia("digitos_permutados", tareas, 64, 10)


def digitos_partidos() -> Secuencia:
    """Cinco tareas de dos clases (0-1, 2-3, …). Cada tarea mira solo sus dos salidas (una cabeza por tarea)."""
    xe, ye, xp, yp = _digitos()
    tareas = []
    for a in range(0, 10, 2):
        me, mp = (ye == a) | (ye == a + 1), (yp == a) | (yp == a + 1)
        tareas.append(Subtarea(torch.tensor(xe[me]), torch.tensor(ye[me]), torch.tensor(xp[mp]), torch.tensor(yp[mp]), [a, a + 1]))
    return Secuencia("digitos_partidos", tareas, 64, 10)


def lunas_rotadas(angulos=(0, 60, 120), n: int = 1500, ruido: float = 0.2, semilla: int = 1234) -> Secuencia:
    """Dos lunas que se giran en cada tarea: la frontera de decisión cambia de forma controlada."""
    from sklearn.datasets import make_moons

    tareas = []
    for k, ang in enumerate(angulos):
        x, y = make_moons(n, noise=ruido, random_state=semilla + k)
        t = math.radians(ang)
        R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
        x = (x - x.mean(0)) @ R.T
        corte = int(n * 0.7)
        tareas.append(Subtarea(*(torch.tensor(a) for a in (x[:corte].astype(np.float32), y[:corte], x[corte:].astype(np.float32), y[corte:]))))
    return Secuencia("lunas_rotadas", tareas, 2, 2)


def continuo(metodo: str, sec: Secuencia, semilla: int, cfg: Config | None = None) -> Resultado:
    """Entrena las tareas en orden con la misma red. ``matriz[i][j]`` = exactitud en la tarea ``j`` justo después de
    entrenar la ``i``. Olvido = media de (lo que valía la tarea al aprenderla − lo que vale al final)."""
    cfg = cfg or Config()
    e = Entrenador(metodo, sec.entradas, sec.salidas, "clasificacion", cfg, semilla)
    matriz = []
    for t in sec.tareas:
        e.fijar_datos(t.x_entreno, t.y_entreno, t.clases)
        e.entrenar_tarea()
        matriz.append([e.evaluar_prueba(u.x_prueba, u.y_prueba, u.clases)[1] for u in sec.tareas])
    r = e.resultado()
    r.matriz = matriz
    T = len(sec.tareas)
    r.metrica_prueba = float(np.mean(matriz[-1]))
    r.perdida_prueba = olvido(matriz) if T > 1 else 0.0  # aquí guarda el olvido
    return r


def olvido(matriz) -> float:
    T = len(matriz)
    return float(np.mean([matriz[j][j] - matriz[T - 1][j] for j in range(T - 1)]))


# --- correr, comparar, criterios ---------------------------------------------------------------------------------


def correr(tarea, metodos, semillas, cfg: Config | None = None) -> dict[str, list[Resultado]]:
    return {m: [entrenar(m, tarea, s, cfg) for s in semillas] for m in metodos}


def correr_continuo(sec: Secuencia, metodos, semillas, cfg: Config | None = None) -> dict[str, list[Resultado]]:
    return {m: [continuo(m, sec, s, cfg) for s in semillas] for m in metodos}


def holm(pvalores: list[float]) -> list[float]:
    """Corrección de Holm-Bonferroni para comparaciones múltiples."""
    orden = np.argsort(pvalores)
    aj, maximo = [0.0] * len(pvalores), 0.0
    for rango, i in enumerate(orden):
        maximo = max(maximo, min(1.0, (len(pvalores) - rango) * pvalores[i]))
        aj[i] = maximo
    return aj


def comparar(a: list[float], b: list[float]) -> dict:
    """Diferencia a − b emparejada por semilla: media, intervalo de confianza del 95 % y valor p (prueba t)."""
    from scipy import stats

    d = np.array(a) - np.array(b)
    n = len(d)
    if n < 2 or np.allclose(d, d[0]):
        return {"dif": float(d.mean()), "ic": (float(d.mean()),) * 2, "p": 1.0 if np.allclose(d, 0) else 0.0}
    se = d.std(ddof=1) / math.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    return {"dif": float(d.mean()), "ic": (float(d.mean() - h), float(d.mean() + h)), "p": float(stats.ttest_rel(a, b).pvalue)}


def contra_comparadores(res: dict[str, list[Resultado]], metodo: str = "rnra", mayor_mejor: bool = True) -> dict[str, dict]:
    """``metodo`` contra cada comparador, con Holm sobre todas las comparaciones."""
    base = [m for m in res if m != metodo and m in COMPARADORES + ["rnra_aleatorio"]]
    sal = {m: comparar([r.metrica_prueba for r in res[metodo]], [r.metrica_prueba for r in res[m]]) for m in base}
    aj = holm([sal[m]["p"] for m in base])
    for m, p in zip(base, aj):
        sal[m]["p_holm"] = p
    return sal


def criterios(res: dict[str, list[Resultado]], metodo: str = "rnra", mayor_mejor: bool = True) -> dict:
    """Los umbrales de la sección 6, fijados antes de ejecutar. Solo cuentan los comparadores con el mismo presupuesto
    de conexiones (se excluye el MLP grande). Devuelve cada criterio con su valor y si se cumple (``ok``)."""
    r = res[metodo]
    media = lambda m, f: float(np.mean([f(x) for x in res[m]]))  # noqa: E731
    cone = media(metodo, lambda x: x.conexiones)
    elegibles = [m for m in COMPARADORES if m in res and media(m, lambda x: x.conexiones) <= cone + 1]
    signo = 1 if mayor_mejor else -1
    mejor = max(elegibles, key=lambda m: signo * media(m, lambda x: x.metrica_prueba))
    dif = signo * (media(metodo, lambda x: x.metrica_prueba) - media(mejor, lambda x: x.metrica_prueba))
    out = {"mejor_comparador": mejor, "dif_precision": dif,
           "precision": {"valor": dif, "ok": dif >= (0.02 if mayor_mejor else 0.0)},
           "parametros": {"valor": media(metodo, lambda x: x.parametros) - media(mejor, lambda x: x.parametros),
                          "ok": media(metodo, lambda x: x.parametros) <= media(mejor, lambda x: x.parametros) * 1.05},
           "estabilidad": {"valor": media(metodo, lambda x: x.ciclos), "ok": media(metodo, lambda x: x.ciclos) == 0}}
    if mayor_mejor:
        objetivo = 0.97 * max(np.mean([x.curva[-1][2] for x in res[m]]) for m in elegibles + [metodo])
        fr = np.mean([x.flops_hasta(objetivo) for x in r])
        fb = np.mean([x.flops_hasta(objetivo) for x in res[mejor]])
        ahorro = 1 - fr / fb if np.isfinite(fr) and np.isfinite(fb) else (1.0 if np.isfinite(fr) else float("nan"))
        out["eficiencia"] = {"valor": float(ahorro), "ok": bool(np.isfinite(fr) and fr <= 0.8 * fb), "objetivo": float(objetivo)}
    return out


def especializacion(entrenador: Entrenador, x: torch.Tensor) -> float:
    """H3: cuánto se concentra la actividad de cada neurona activa en pocas regiones de la entrada
    (0 = reparte igual en todas, 1 = vive en una sola). Se compara contra el control aleatorio."""
    reg = entrenador.reg.asignar(x)
    k = entrenador.reg.k
    with torch.no_grad():
        entrenador.m(x, guardar=True)
        puntajes = []
        for l, h in enumerate(entrenador.m._h):
            act = (entrenador.m.activa(l) > 0).nonzero().flatten()
            for i in act.tolist():
                a = torch.zeros(k).index_add_(0, reg, h[:, i].abs()) / torch.bincount(reg, minlength=k).clamp(min=1)
                if a.sum() > 0:
                    p = a / a.sum()
                    ent = -(p[p > 0] * p[p > 0].log()).sum() / math.log(k) if k > 1 else 0.0
                    puntajes.append(1 - float(ent))
    return float(np.mean(puntajes)) if puntajes else 0.0


# --- informe -----------------------------------------------------------------------------------------------------


def _ms(v: list[float], pct: bool) -> str:
    m, s = float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    return f"{m * 100:.1f} ± {s * 100:.1f} %" if pct else f"{m:.3f} ± {s:.3f}"


def tabla(res: dict[str, list[Resultado]], pct: bool = True, referencia: str = "mlp_pequeno") -> str:
    """Tabla en Markdown: métrica de prueba, parámetros activos, conexiones, FLOPs y ciclos estructurales."""
    filas = ["| Método | Prueba | Parámetros activos | Conexiones | FLOPs (×10⁶) | Ciclos |", "|---|---|---|---|---|---|"]
    for m, rs in res.items():
        filas.append(f"| `{m}` | {_ms([r.metrica_prueba for r in rs], pct)} | {np.mean([r.parametros for r in rs]):.0f} | "
                     f"{np.mean([r.conexiones for r in rs]):.0f} | {np.mean([r.flops for r in rs]) / 1e6:.0f} | "
                     f"{np.mean([r.ciclos for r in rs]):.1f} |")
    return "\n".join(filas)


def tabla_comparacion(res: dict[str, list[Resultado]], metodo: str = "rnra", pct: bool = True) -> str:
    c = contra_comparadores(res, metodo)
    esc = 100 if pct else 1
    filas = [f"| `{metodo}` contra | Diferencia | IC 95 % | p (Holm) |", "|---|---|---|---|"]
    for m, d in c.items():
        filas.append(f"| `{m}` | {d['dif'] * esc:+.2f}{' pp' if pct else ''} | [{d['ic'][0] * esc:+.2f}, {d['ic'][1] * esc:+.2f}] | {d['p_holm']:.3f} |")
    return "\n".join(filas)
