"""Cuaderno 06 · Entornos de refuerzo: escribir código, empujar cubos en equipo y ReroutingNetwork.

1. **Escribir código** (``RawPythonEnv``, celda 6): el agente elige caracteres uno por uno (letras, dígitos,
   ``+ - = " ( ) _``, la palabra ``print`` y fin de línea) y un intérprete pequeño ejecuta cada línea. Recompensa
   (``calculate_custom_reward``): −0,005 por paso, −1 por línea con error, +0,5 por línea válida, +5 si la salida es
   el objetivo. Se entrena con PPO (``stable-baselines3``), con los mismos ajustes del cuaderno.
   - Fallo: el objetivo era «hola mundo», pero **el espacio no está entre los caracteres**: es imposible.
   - Fallo: la demostración usaba ``reset``/``step`` del entorno vectorizado con la forma de Gymnasium (5 valores)
     y se caía.
   - Arreglo: objetivo alcanzable («hola»: ``print("hola")`` son 10 acciones).
2. **Empujar cubos con dos cuerpos** (``MultiBodyCubePushEnv``, celda 18): en una cuadrícula de 10×10, dos cuerpos
   deben empujar un cubo **a la vez y en la misma dirección** hasta la esquina. Recompensa: −0,1 por paso, +0,5 por
   acercar el cubo, +10 al llegar.
   - Fallo: no había límite de pasos; un episodio sin éxito no terminaba nunca. Aquí: 200 pasos.
   - Arreglo: posiciones normalizadas a [0, 1] (las redes aprenden mejor así) y entrenamiento más largo.
3. **ReroutingNetwork** (celda 36): REINFORCE en un problema de juguete de 5 estados y 2 acciones, 20 pasos por
   episodio. Como es tan pequeño, la mejor política posible se calcula exacta (programación dinámica).
   - Tal cual: cada acción se juzga por la recompensa de **todo** el episodio, sin línea base.
   - Arreglo (REINFORCE estándar): cada acción se juzga por lo que viene **después** de ella (recompensa futura),
     menos el promedio.

Rivales: un agente al azar y, en ReroutingNetwork, la política óptima exacta.
"""

from __future__ import annotations

import numpy as np


# ================================================================ 1. Escribir código
def entorno_codigo(objetivo: str = "hola", max_tokens: int = 20):
    import gymnasium as gym
    from gymnasium import spaces

    letras, digitos = "abcdefghijklmnopqrstuvwxyz", "0123456789"
    acciones = list(letras) + list(digitos) + ["+", "-", "=", '"', "(", ")", "_"] + ["print"] + ["_EOS_"]

    class RawPythonEnv(gym.Env):
        def __init__(self):
            self.acciones, self.objetivo, self.max_tokens = acciones, objetivo, max_tokens
            self.action_space = spaces.Discrete(len(acciones))
            self.observation_space = spaces.Box(0, 1, (5 * len(acciones) + len(letras) + 2,), np.float32)

        def _obs(self):
            o = np.zeros(5 * len(acciones), np.float32)
            for i, tok in enumerate(self.tokens[-5:]):
                o[i * len(acciones) + acciones.index(tok)] = 1
            vars_ = np.array([v in self.vars for v in letras], np.float32)
            return np.concatenate([o, vars_, [len(self.salida) / 100.0, float(self.objetivo in self.salida)]]).astype(np.float32)

        def _valor(self, s):
            s = s.strip()
            if s.startswith('"') and s.endswith('"') and len(s) >= 2:
                return s[1:-1]
            if s.isdigit():
                return int(s)
            return self.vars.get(s)

        def _interpretar(self, linea):
            """El intérprete del cuaderno: print("..."), print(a), print(a+b), print(a-b), a="...", a=1, a=b+c…"""
            def expr(e):
                for op in "+-":
                    if op in e:
                        p = e.split(op)
                        if len(p) != 2:
                            return None
                        a, b = self._valor(p[0]), self._valor(p[1])
                        if a is None or b is None:
                            return None
                        try:
                            return a + b if op == "+" else (a - b if isinstance(a, int) and isinstance(b, int) else None)
                        except TypeError:
                            return None
                return self._valor(e)

            if linea.startswith("print(") and linea.endswith(")"):
                v = expr(linea[6:-1])
                return (None, True) if v is None else (str(v), False)
            if "=" in linea:
                nombre, valor = linea.split("=", 1)
                v = expr(valor)
                if nombre.strip() in letras and v is not None:
                    self.vars[nombre.strip()] = v
                    return "", False
            return None, True

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.tokens, self.salida, self.vars, self.pasos = [], "", {}, 0
            return self._obs(), {}

        def step(self, a):
            tok = acciones[a]
            self.pasos += 1
            r = -0.005
            if tok == "_EOS_":
                salida, error = self._interpretar("".join(self.tokens))
                if error:
                    r -= 1.0
                else:
                    self.salida += salida
                    r += 0.5
                    if self.salida == self.objetivo:
                        r += 5.0
                    elif self.objetivo in self.salida:
                        r += 2.0
                self.tokens = []
            else:
                self.tokens.append(tok)
                if len(self.tokens) >= self.max_tokens:
                    r -= 0.5
            terminado = self.objetivo in self.salida
            cortado = (len(self.tokens) >= 2 * self.max_tokens or len(self.salida) > 2 * len(self.objetivo) + 10
                       or self.pasos >= 5 * self.max_tokens)
            return self._obs(), r, terminado, cortado and not terminado, {"exito": terminado}

    return RawPythonEnv()


# ================================================================ 2. Cubos
def entorno_cubos(normalizar: bool = False, max_pasos: int = 200):
    import gymnasium as gym
    from gymnasium import spaces

    lado = 10
    mov = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0), 4: (0, 0)}

    class MultiBodyCubePushEnv(gym.Env):
        def __init__(self):
            self.action_space = spaces.MultiDiscrete([5, 5])
            self.observation_space = spaces.Box(0, 1 if normalizar else lado - 1, (8,), np.float32)
            self.meta = np.array([lado - 1, lado - 1])

        def _obs(self):
            o = np.concatenate([self.c1, self.c2, self.cubo, self.meta]).astype(np.float32)
            return o / (lado - 1) if normalizar else o

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            while True:
                self.c1, self.c2, self.cubo = (self.np_random.integers(0, lado, 2) for _ in range(3))
                if not (np.array_equal(self.c1, self.c2) or np.array_equal(self.c1, self.cubo)
                        or np.array_equal(self.c2, self.cubo) or np.array_equal(self.cubo, self.meta)):
                    break
            self.dist = np.linalg.norm(self.cubo - self.meta)
            self.pasos = 0
            return self._obs(), {}

        def step(self, accion):
            a1, a2 = int(accion[0]), int(accion[1])
            p1 = np.clip(self.c1 + mov[a1], 0, lado - 1)
            p2 = np.clip(self.c2 + mov[a2], 0, lado - 1)
            q1, q2 = np.array_equal(p1, self.cubo), np.array_equal(p2, self.cubo)
            movido = False
            if q1 and q2 and a1 == a2 and a1 != 4:  # los dos empujan juntos en la misma dirección
                nuevo = self.cubo + mov[a1]
                if (0 <= nuevo).all() and (nuevo < lado).all():
                    self.c1 = self.c2 = self.cubo.copy()
                    self.cubo = nuevo
                    movido = True
            if not movido:
                self.c1 = self.c1 if q1 else p1
                self.c2 = self.c2 if q2 else p2
            d = np.linalg.norm(self.cubo - self.meta)
            r = -0.1 + (self.dist - d) * 0.5
            self.dist = d
            self.pasos += 1
            exito = bool(np.array_equal(self.cubo, self.meta))
            if exito:
                r = 10.0
            return self._obs(), r, exito, (self.pasos >= max_pasos) and not exito, {"exito": exito}

    return MultiBodyCubePushEnv()


# ================================================================ evaluación común (PPO de stable-baselines3)
def _evaluar(env, politica, episodios: int, semilla: int) -> dict:
    rng = np.random.default_rng(semilla)
    exitos, largos = 0, []
    for e in range(episodios):
        obs, _ = env.reset(seed=semilla * 1000 + e)
        for t in range(10_000):
            obs, _, term, trunc, info = env.step(politica(obs, rng))
            if term or trunc:
                break
        exitos += bool(info.get("exito"))
        largos.append(t + 1)
    return {"exito": exitos / episodios, "pasos": float(np.mean(largos))}


def ppo(env_fn, pasos: int, semilla: int, **ajustes):
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    venv = make_vec_env(env_fn, n_envs=1, seed=semilla)
    m = PPO("MlpPolicy", venv, seed=semilla, verbose=0, device="cpu", **ajustes)
    m.learn(total_timesteps=pasos)
    return lambda obs, rng: m.predict(obs, deterministic=False)[0]


AJUSTES_CODIGO = dict(learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                      clip_range=0.2, ent_coef=0.01, policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])))
AJUSTES_CUBOS = dict(learning_rate=1e-3, n_steps=1024, batch_size=64, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                     ent_coef=0.01)


def competir_codigo(semillas=(0, 1, 2), pasos: int = 100_000, episodios: int = 50, aviso=print) -> dict:
    r = {}
    for nombre, objetivo in (("hola mundo (tal cual: sin espacio, imposible)", "hola mundo"), ("hola (arreglo)", "hola")):
        al_azar = _evaluar(entorno_codigo(objetivo), lambda o, g: g.integers(0, 45), episodios, 0)
        filas = []
        for s in semillas:
            pol = ppo(lambda o=objetivo: entorno_codigo(o), pasos, s, **AJUSTES_CODIGO)
            filas.append(_evaluar(entorno_codigo(objetivo), pol, episodios, s))
            aviso(f"código {objetivo!r} semilla {s}: {filas[-1]}")
        r[nombre] = {"al_azar": al_azar, "ppo": filas}
    return r


def competir_cubos(semillas=(0, 1, 2), pasos: int = 100_000, pasos_arreglo: int = 300_000, episodios: int = 50,
                   aviso=print) -> dict:
    r = {"al_azar": _evaluar(entorno_cubos(), lambda o, g: g.integers(0, 5, 2), episodios, 0)}
    for nombre, normal, n in (("ppo_cuaderno", False, pasos), ("ppo_arreglado", True, pasos_arreglo)):
        filas = []
        for s in semillas:
            pol = ppo(lambda nm=normal: entorno_cubos(nm), n, s, **AJUSTES_CUBOS)
            filas.append(_evaluar(entorno_cubos(normal), pol, episodios, s))
            aviso(f"cubos {nombre} semilla {s}: {filas[-1]}")
        r[nombre] = filas
    return r


# ================================================================ 3. ReroutingNetwork
RECOMPENSAS = {(0, 0): 1, (1, 0): 2, (2, 1): 1, (3, 1): 3, (4, 1): 5}
TRANSICIONES = {(0, 0): 1, (1, 0): 2, (2, 1): 3, (3, 1): 0, (4, 1): 1}


def optimo_rerouting(pasos: int = 20) -> float:
    """Retorno medio de la mejor política (programación dinámica), empezando en un estado al azar."""
    v = np.zeros(5)
    for _ in range(pasos):
        v = np.array([max(RECOMPENSAS.get((s, a), 0) + v[TRANSICIONES.get((s, a), s)] for a in (0, 1)) for s in range(5)])
    return float(v.mean())


def rerouting(modo: str, semilla: int = 0, episodios: int = 500, pasos: int = 20, lr: float = 0.01) -> float:
    """``modo``: "cuaderno" (retorno total, sin línea base) o "arreglado" (recompensa futura menos el promedio).
    Devuelve el retorno medio de la política aprendida en 200 episodios de prueba."""
    import torch
    from torch import nn

    torch.manual_seed(semilla)
    rng = np.random.default_rng(semilla)
    red = nn.Sequential(nn.Linear(5, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2))
    opt = torch.optim.Adam(red.parameters(), lr=lr)
    ojo = torch.eye(5)

    def episodio(aprender):
        s, logps, rs = int(rng.integers(5)), [], []
        for _ in range(pasos):
            dist = torch.distributions.Categorical(logits=red(ojo[s]))
            a = dist.sample()
            logps.append(dist.log_prob(a))
            rs.append(RECOMPENSAS.get((s, int(a)), 0))
            s = TRANSICIONES.get((s, int(a)), s)
        return logps, rs

    for _ in range(episodios):
        logps, rs = episodio(True)
        if modo == "cuaderno":
            perdida = -torch.stack(logps).sum() * sum(rs)
        else:
            futuro = torch.tensor(np.cumsum(rs[::-1])[::-1].copy(), dtype=torch.float32)
            ventaja = (futuro - futuro.mean()) / (futuro.std() + 1e-6)
            perdida = -(torch.stack(logps) * ventaja).sum()
        opt.zero_grad()
        perdida.backward()
        opt.step()
    with torch.no_grad():
        return float(np.mean([sum(episodio(False)[1]) for _ in range(200)]))


def competir_rerouting(semillas=(0, 1, 2, 3, 4)) -> dict:
    rng = np.random.default_rng(0)

    def al_azar():
        s, total = int(rng.integers(5)), 0
        for _ in range(20):
            a = int(rng.integers(2))
            total += RECOMPENSAS.get((s, a), 0)
            s = TRANSICIONES.get((s, a), s)
        return total

    return {"optimo": optimo_rerouting(), "al_azar": float(np.mean([al_azar() for _ in range(2000)])),
            "cuaderno": [rerouting("cuaderno", s) for s in semillas],
            "arreglado": [rerouting("arreglado", s) for s in semillas]}


# ================================================================ informe
def _pct(filas, k="exito"):
    a = np.array([f[k] for f in filas])
    return f"{a.mean():.0%}" + (f" ± {a.std():.0%}" if len(a) > 1 else "")


def _num(x, dec=1):
    return f"{x:,.{dec}f}".replace(",", " ").replace(".", ",")


def informe(codigo: dict, cubos: dict, rer: dict, pasos_codigo: int, pasos_cubos: tuple, episodios: int = 30) -> str:
    sem_cod = len(next(iter(codigo.values()))["ppo"])
    l = ["# Cuaderno 06: entornos de refuerzo", "",
         f"Cada agente se prueba en {episodios} episodios nuevos por semilla. «Éxito» = llegó a la meta.", "",
         f"## 1. Escribir código con PPO ({_num(pasos_codigo, 0)} pasos de entrenamiento, {sem_cod} semillas)", "",
         "| Objetivo | Agente al azar | PPO (ajustes del cuaderno) |", "|---|---|---|"]
    for nombre, v in codigo.items():
        l.append(f"| {nombre} | {v['al_azar']['exito']:.0%} | {_pct(v['ppo'])} |")
    l += ["", f"## 2. Empujar el cubo entre dos (cuadrícula 10×10, máximo 200 pasos, {len(cubos['ppo_cuaderno'])} semillas)", "",
          "| Agente | Éxito | Pasos por episodio |", "|---|---|---|",
          f"| al azar | {cubos['al_azar']['exito']:.0%} | {cubos['al_azar']['pasos']:.0f} |"]
    for nombre, texto, n in (("ppo_cuaderno", "PPO tal cual", pasos_cubos[0]),
                             ("ppo_arreglado", "PPO con posiciones normalizadas", pasos_cubos[1])):
        f = cubos[nombre]
        l.append(f"| {texto} ({_num(n, 0)} pasos) | {_pct(f)} | {np.mean([x['pasos'] for x in f]):.0f} |")
    a, b = np.array(rer["cuaderno"]), np.array(rer["arreglado"])
    l += ["", f"## 3. ReroutingNetwork (5 estados, 20 pasos por episodio, {len(a)} semillas)", "",
          "| Política | Recompensa por episodio |", "|---|---|",
          f"| la mejor posible (calculada exacta) | {_num(rer['optimo'])} |",
          f"| REINFORCE tal cual (retorno total, sin línea base) | {_num(a.mean())} ± {_num(a.std())} |",
          f"| REINFORCE estándar (recompensa futura menos el promedio) | {_num(b.mean())} ± {_num(b.std())} |",
          f"| al azar | {_num(rer['al_azar'])} |", ""]
    return "\n".join(l)
