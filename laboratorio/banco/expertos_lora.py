"""Cuaderno 14 · Tronco enrutador + mini-expertos LoRA por oficio (electricidad, agricultura, mecánica, redes).

La idea: un LLM «maestro» etiqueta preguntas por dominio; con esas etiquetas se entrena un **router troncal**
pequeño (milisegundos, sin gastar API), que decide qué **mini-experto** (un adaptador LoRA sobre un modelo base
pequeño) responde. El maestro también escribe pares pregunta → respuesta para entrenar cada experto.

**Quién hace de maestro** (se elige solo, en este orden):
1. **DeepSeek**, si la variable de entorno ``DEEPSEEK_API_KEY`` tiene una llave (en Colab: Secretos → ``DEEPSEEK_API_KEY``).
2. **Un modelo pequeño de Hugging Face** (por defecto ``Qwen/Qwen2.5-1.5B-Instruct``), si ``transformers`` está
   instalado. Cambia el modelo con la variable ``MAESTRO_HF`` (por ejemplo ``Qwen/Qwen2.5-0.5B-Instruct``).
3. **Reglas por palabras clave**, sin internet (sirve para las pruebas y para arrancar).

Nunca escribas la llave en el código: el valor por defecto es vacío y en ese caso se pasa al siguiente maestro.

**Qué se compara** (para saber si vale la pena):
- Router: tu ``RouterTroncal`` (red de 2 capas sobre embeddings) contra una regresión logística (industria) y
  contra preguntar directo al maestro.
- Respuestas: modelo base sin ajustar · **un solo LoRA con todos los dominios** (el rival clave: muchas veces
  iguala a los expertos) · **router + un LoRA por dominio** (tu idea) · **oráculo** (el LoRA correcto siempre: el
  techo de la idea). Métrica: perplejidad de las respuestas de prueba (menos es mejor).

Trabajos parecidos: Branch-Train-MiX (Meta, 2024), mezclas de expertos LoRA (MoR-LoRA, RAMoLE, LoRA-Mixer, MoLoRA).
Lo propio aquí: oficios prácticos en español, y medirlo contra el LoRA único y el oráculo.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import numpy as np

DOMINIOS = ["electricidad", "agricultura", "mecanica", "redes"]

# Preguntas de ejemplo (escritas a mano) para probar el router sin internet. El cuaderno 14 genera muchas más.
SEMILLA = {
    "electricidad": [
        "¿Qué calibre de cable uso para un motor trifásico de 5 HP a 220 V?",
        "¿Cómo calculo la caída de tensión en un circuito de 50 metros?",
        "¿Por qué salta el disyuntor diferencial cuando enciendo la ducha eléctrica?",
        "¿Cuántos tomacorrientes puedo poner en un circuito de 20 amperios?",
        "¿Cómo conecto un interruptor de tres vías para una escalera?",
        "¿Qué diferencia hay entre el neutro y la tierra en una instalación?",
        "¿Cómo dimensiono un banco de baterías para paneles solares?",
        "¿Qué protección necesita un tablero eléctrico de una casa?",
        "¿Cómo mido la resistencia de la puesta a tierra con un telurómetro?",
        "¿Por qué se calienta un contactor y cómo lo reviso?",
        "¿Qué potencia consume un aire acondicionado de 12000 BTU?",
        "¿Cómo arranco un motor con variador de frecuencia?",
    ],
    "agricultura": [
        "¿Cuál es el pH ideal del suelo para cultivar cacao?",
        "¿Cada cuánto riego el maíz en época seca?",
        "¿Cómo controlo la sigatoka negra en el banano?",
        "¿Qué abono orgánico sirve para hortalizas de hoja?",
        "¿Cuándo es la mejor época para sembrar arroz en la costa?",
        "¿Cómo preparo un compost con restos de cosecha?",
        "¿Qué distancia de siembra uso para plantas de café?",
        "¿Cómo sé si mi cultivo de papa tiene tizón tardío?",
        "¿Qué cantidad de nitrógeno necesita una hectárea de maíz?",
        "¿Cómo injerto un árbol de mango?",
        "¿Qué hago contra la broca del café?",
        "¿Cómo mejoro un suelo arcilloso que se encharca?",
    ],
    "mecanica": [
        "¿Cada cuántos kilómetros cambio el aceite de un motor diésel?",
        "¿Por qué mi carro pierde potencia en subidas?",
        "¿Cómo reviso la presión de compresión de un cilindro?",
        "¿Qué significa que el motor queme aceite y salga humo azul?",
        "¿Cómo purgo el sistema de frenos hidráulicos?",
        "¿Por qué vibra el volante a alta velocidad?",
        "¿Cómo ajusto la holgura de las válvulas?",
        "¿Qué causa que la batería del auto se descargue de noche?",
        "¿Cómo cambio la correa de distribución?",
        "¿Por qué se recalienta el motor si el radiador está lleno?",
        "¿Qué torque llevan las tuercas de una rueda?",
        "¿Cómo diagnostico un inyector tapado?",
    ],
    "redes": [
        "¿Cómo configuro una VLAN para separar voz y datos en un switch?",
        "¿Qué máscara de subred uso para 50 equipos?",
        "¿Por qué mi wifi es lento aunque tengo buena señal?",
        "¿Cómo abro un puerto en el router para una cámara IP?",
        "¿Qué diferencia hay entre un switch administrable y uno no administrable?",
        "¿Cómo armo un cable de red categoría 6 con norma T568B?",
        "¿Cómo configuro un DHCP en un servidor Linux?",
        "¿Qué es el ping y cómo veo la pérdida de paquetes?",
        "¿Cómo enlazo dos edificios con antenas punto a punto?",
        "¿Cómo protejo mi red wifi de intrusos?",
        "¿Qué es un firewall y cómo lo configuro en el router?",
        "¿Cómo conecto dos routers en la misma red?",
    ],
}

# Preguntas que mezclan dominios (para ver qué hace el router con ellas).
MIXTAS = [
    ("¿Qué bomba eléctrica y qué cable necesito para regar 2 hectáreas?", ("agricultura", "electricidad")),
    ("¿Cómo monitoreo por wifi la temperatura de mi invernadero?", ("agricultura", "redes")),
    ("¿Por qué el alternador del tractor no carga la batería?", ("mecanica", "electricidad")),
]

PALABRAS = {
    "electricidad": ["cable", "voltaje", "tensión", " v ", "amperio", "disyuntor", "breaker", "tomacorriente", "interruptor",
                     "neutro", "tierra", "batería", "solar", "tablero", "motor trifásico", "contactor", "potencia", "variador"],
    "agricultura": ["suelo", "riego", "regar", "cultivo", "siembra", "sembrar", "abono", "compost", "cacao", "maíz",
                    "banano", "café", "papa", "arroz", "plaga", "hectárea", "injerto", "cosecha", "invernadero"],
    "mecanica": ["motor", "aceite", "carro", "auto", "freno", "volante", "válvula", "correa", "radiador", "rueda",
                 "inyector", "cilindro", "tractor", "alternador", "embrague", "diésel"],
    "redes": ["wifi", "router", "switch", "vlan", "subred", "ip", "puerto", "dhcp", "ping", "paquete", "firewall",
              "antena", "cable de red", "red "],
}


# ================================================================ maestros (etiquetan y generan)
@dataclass
class Maestro:
    nombre: str
    llamar: object  # función (sistema, usuario) -> texto
    notas: str = ""

    def etiquetar(self, pregunta: str) -> str:
        sistema = ("Eres un clasificador. Responde ÚNICAMENTE con una de estas categorías, sin explicar: "
                   + ", ".join(DOMINIOS) + ".")
        return normalizar_dominio(self.llamar(sistema, pregunta))

    def generar_pares(self, dominio: str, n: int = 20, enfoque: str = "") -> list[dict]:
        """Pide ``n`` pares pregunta → respuesta experta del dominio, en español, como líneas JSON.

        ``enfoque`` (p. ej. «seguridad», «diagnóstico de fallas») cambia el pedido en cada tanda, para que el
        maestro no repita las mismas preguntas.
        """
        sistema = ("Eres un experto que escribe material de entrenamiento en español claro. Devuelve SOLO líneas "
                   'JSON, una por ejemplo, con las claves "pregunta" y "respuesta".')
        usuario = (f"Escribe {n} preguntas prácticas y distintas de un técnico de {dominio}, cada una con una respuesta "
                   "experta de 3 a 6 frases que explique el porqué.")
        if enfoque:
            usuario += f" Todas sobre este tema: {enfoque}."
        pares = []
        for linea in self.llamar(sistema, usuario).splitlines():
            linea = linea.strip().strip(",")
            if not linea.startswith("{"):
                continue
            try:
                d = json.loads(linea)
            except json.JSONDecodeError:
                continue
            if d.get("pregunta") and d.get("respuesta"):
                pares.append({"pregunta": d["pregunta"], "respuesta": d["respuesta"], "dominio": dominio})
        return pares


ENFOQUES = ["conceptos básicos", "seguridad", "diagnóstico de fallas", "instalación", "mantenimiento",
            "herramientas y medición", "costos y ahorro", "normas y buenas prácticas", "casos difíciles",
            "errores comunes de principiantes", "eficiencia", "elegir materiales o equipos"]


def _clave(pregunta: str) -> str:
    return re.sub(r"[^a-záéíóúüñ0-9 ]", "", pregunta.lower()).strip()


def generar_datos(maestro: Maestro, dominios=DOMINIOS, por_dominio: int = 200, tanda: int = 20,
                  intentos_por_tanda: int = 3, aviso=print) -> list[dict]:
    """Genera ``por_dominio`` pares **sin repetir preguntas**, cambiando el enfoque en cada tanda.

    Se detiene con un error claro si el maestro no sabe generar (las reglas solo etiquetan), y deja de pedir
    si varias tandas seguidas no traen preguntas nuevas (respuestas mal formadas o repetidas), para no quedar
    en un ciclo infinito.
    """
    if maestro.nombre == "reglas":
        raise RuntimeError("El maestro de reglas solo etiqueta, no escribe preguntas. Guarda DEEPSEEK_API_KEY en los "
                           "Secretos de Colab o usa una GPU para el modelo de Hugging Face (MAESTRO_HF).")
    pares, vistas = [], set()
    for d in dominios:
        cuantas, sin_nuevas, k = 0, 0, 0
        while cuantas < por_dominio and sin_nuevas < intentos_por_tanda:
            nuevas = 0
            for p in maestro.generar_pares(d, tanda, ENFOQUES[k % len(ENFOQUES)]):
                c = _clave(p["pregunta"])
                if c and c not in vistas and cuantas < por_dominio:
                    vistas.add(c)
                    pares.append(p)
                    cuantas += 1
                    nuevas += 1
            k += 1
            sin_nuevas = 0 if nuevas else sin_nuevas + 1
            aviso(f"{d}: {cuantas}")
        if cuantas < por_dominio:
            aviso(f"Aviso: {d} quedó con {cuantas} de {por_dominio} (el maestro dejó de dar preguntas nuevas).")
    return pares


def separar(pares: list[dict], prueba: float = 0.2, semilla: int = 0) -> tuple[list[dict], list[dict]]:
    """Separa entrenamiento y prueba **sin preguntas repetidas entre las dos partes** (por si el archivo trae
    duplicados de una corrida vieja), y en la misma proporción por dominio."""
    import random

    unicos, vistas = [], set()
    for p in pares:
        c = _clave(p["pregunta"])
        if c not in vistas:
            vistas.add(c)
            unicos.append(p)
    rng = random.Random(semilla)
    entreno, prueba_ = [], []
    for d in sorted({p["dominio"] for p in unicos}):
        grupo = [p for p in unicos if p["dominio"] == d]
        rng.shuffle(grupo)
        corte = int(round((1 - prueba) * len(grupo)))
        entreno += grupo[:corte]
        prueba_ += grupo[corte:]
    return entreno, prueba_


def normalizar_dominio(texto: str) -> str:
    t = texto.lower()
    t = t.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
    for d in DOMINIOS:
        if d in t:
            return d
    sinonimos = {"electric": "electricidad", "agri": "agricultura", "mecan": "mecanica", "red": "redes",
                 "telecom": "redes"}
    for k, v in sinonimos.items():
        if k in t:
            return v
    return "desconocido"


def maestro_reglas() -> Maestro:
    def llamar(sistema: str, usuario: str) -> str:
        t = " " + usuario.lower() + " "
        puntos = {d: sum(t.count(p) for p in ps) for d, ps in PALABRAS.items()}
        return max(puntos, key=puntos.get) if max(puntos.values()) > 0 else "desconocido"

    return Maestro("reglas", llamar, "palabras clave (sin internet; solo para probar)")


def maestro_deepseek(llave: str | None = None, modelo: str = "deepseek-chat") -> Maestro | None:
    llave = llave if llave is not None else os.environ.get("DEEPSEEK_API_KEY", "")
    if not llave:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    cliente = OpenAI(api_key=llave, base_url="https://api.deepseek.com")

    def llamar(sistema: str, usuario: str) -> str:
        r = cliente.chat.completions.create(model=modelo, temperature=0.0,
                                            messages=[{"role": "system", "content": sistema},
                                                      {"role": "user", "content": usuario}])
        return r.choices[0].message.content or ""

    return Maestro("deepseek", llamar, modelo)


def maestro_hf(modelo: str | None = None) -> Maestro | None:
    modelo = modelo or os.environ.get("MAESTRO_HF", "Qwen/Qwen2.5-1.5B-Instruct")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        return None
    try:
        tok = AutoTokenizer.from_pretrained(modelo)
        m = AutoModelForCausalLM.from_pretrained(modelo, torch_dtype="auto",
                                                 device_map="auto" if torch.cuda.is_available() else None)
    except Exception:  # sin internet o modelo inexistente: se pasa al siguiente maestro
        return None

    def llamar(sistema: str, usuario: str) -> str:
        texto = tok.apply_chat_template([{"role": "system", "content": sistema}, {"role": "user", "content": usuario}],
                                        tokenize=False, add_generation_prompt=True)
        x = tok(texto, return_tensors="pt").to(m.device)
        with torch.no_grad():
            # Con muestreo: cada tanda sale distinta (sin él, el mismo pedido da siempre las mismas preguntas).
            y = m.generate(**x, max_new_tokens=1024, do_sample=True, temperature=0.8, top_p=0.95)
        return tok.decode(y[0, x["input_ids"].shape[1]:], skip_special_tokens=True)

    return Maestro("hf", llamar, modelo)


def elegir_maestro(preferir: str | None = None) -> Maestro:
    """DeepSeek si hay llave → modelo de Hugging Face → reglas. ``preferir`` fuerza uno ("deepseek", "hf", "reglas")."""
    opciones = {"deepseek": maestro_deepseek, "hf": maestro_hf, "reglas": maestro_reglas}
    orden = [preferir] if preferir else ["deepseek", "hf", "reglas"]
    for nombre in orden:
        m = opciones[nombre]()
        if m is not None:
            return m
    return maestro_reglas()


# ================================================================ router troncal
def embeddings(textos: list[str], vectorizador=None):
    """Embeddings de oraciones: MiniLM multilingüe si está instalado; si no, TF-IDF de caracteres."""
    try:
        from sentence_transformers import SentenceTransformer

        modelo = vectorizador or SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        return np.asarray(modelo.encode(textos, normalize_embeddings=True), dtype=np.float32), modelo
    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer

        if vectorizador is None or not hasattr(vectorizador, "transform"):
            vectorizador = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1).fit(textos)
        return vectorizador.transform(textos).toarray().astype(np.float32), vectorizador


class RouterTroncal:
    """Tu tálamo: Lineal → ReLU → Dropout → Lineal sobre el embedding de la pregunta. Devuelve el dominio."""

    def __init__(self, dim: int, ocultas: int = 128, semilla: int = 0) -> None:
        import torch
        from torch import nn

        torch.manual_seed(semilla)
        self.red = nn.Sequential(nn.Linear(dim, ocultas), nn.ReLU(), nn.Dropout(0.1), nn.Linear(ocultas, len(DOMINIOS)))

    def entrenar(self, x: np.ndarray, y: list[str], epocas: int = 200, lr: float = 1e-2):
        import torch

        xt, yt = torch.as_tensor(x), torch.as_tensor([DOMINIOS.index(d) for d in y])
        opt = torch.optim.Adam(self.red.parameters(), lr=lr, weight_decay=1e-4)
        self.red.train()
        for _ in range(epocas):
            perdida = torch.nn.functional.cross_entropy(self.red(xt), yt)
            opt.zero_grad(); perdida.backward(); opt.step()
        return self

    def probabilidades(self, x: np.ndarray) -> np.ndarray:
        import torch

        self.red.eval()
        with torch.no_grad():
            return torch.softmax(self.red(torch.as_tensor(x)), 1).numpy()

    def predecir(self, x: np.ndarray, top: int = 1):
        p = self.probabilidades(x)
        orden = np.argsort(-p, 1)[:, :top]
        return [[DOMINIOS[i] for i in fila] for fila in orden] if top > 1 else [DOMINIOS[i] for i in orden[:, 0]]


def comparar_routers(textos_e, y_e, textos_p, y_p, semillas=(0, 1, 2)) -> dict:
    """Acierto del router troncal contra una regresión logística (industria), con los mismos embeddings."""
    from sklearn.linear_model import LogisticRegression

    xe, vec = embeddings(textos_e)
    xp, _ = embeddings(textos_p, vec)
    tronco = [float(np.mean(np.array(RouterTroncal(xe.shape[1], semilla=s).entrenar(xe, y_e).predecir(xp)) == np.array(y_p)))
              for s in semillas]
    log = LogisticRegression(max_iter=2000).fit(xe, y_e)
    return {"router_troncal": tronco, "logistica": float(np.mean(log.predict(xp) == np.array(y_p))),
            "dim": int(xe.shape[1])}


# ================================================================ mini-expertos LoRA (GPU)
@dataclass
class Expertos:
    """Un modelo base con varios adaptadores LoRA cargados a la vez; se cambia de experto sin recargar el modelo."""
    base: str
    modelo: object = None
    tok: object = None
    adaptadores: list[str] = field(default_factory=list)


def entrenar_lora(base: str, pares: list[dict], salida: str, epocas: int = 2, rango: int = 16, lr: float = 2e-4):
    """Ajusta un adaptador LoRA con pares pregunta → respuesta (formato de chat del modelo base)."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
                                             device_map="auto" if torch.cuda.is_available() else None)
    m = get_peft_model(m, LoraConfig(r=rango, lora_alpha=2 * rango, lora_dropout=0.05, task_type="CAUSAL_LM",
                                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr)
    m.train()
    for _ in range(epocas):
        for par in pares:
            x = _codificar(tok, par, m.device)
            perdida = m(**x).loss
            opt.zero_grad(); perdida.backward(); opt.step()
    m.save_pretrained(salida)
    return salida


def _codificar(tok, par: dict, dispositivo):
    """Tokens de pregunta + respuesta; la pérdida solo cuenta la respuesta."""
    import torch

    pregunta = tok.apply_chat_template([{"role": "user", "content": par["pregunta"]}], tokenize=False,
                                       add_generation_prompt=True)
    a = tok(pregunta, add_special_tokens=False)["input_ids"]
    b = tok(par["respuesta"] + (tok.eos_token or ""), add_special_tokens=False)["input_ids"]
    ids = torch.tensor([a + b], device=dispositivo)
    etiquetas = torch.tensor([[-100] * len(a) + b], device=dispositivo)
    return {"input_ids": ids, "labels": etiquetas}


def cargar_expertos(base: str, carpetas: dict[str, str]) -> Expertos:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
                                             device_map="auto" if torch.cuda.is_available() else None)
    nombres = list(carpetas)
    m = PeftModel.from_pretrained(m, carpetas[nombres[0]], adapter_name=nombres[0])
    for n in nombres[1:]:
        m.load_adapter(carpetas[n], adapter_name=n)
    return Expertos(base, m, tok, nombres)


def perplejidad(expertos: Expertos, pares: list[dict], elegir) -> float:
    """``elegir(par)`` devuelve el nombre del adaptador a usar (o None para el modelo base sin ajustar)."""
    import math

    import torch

    total, cuantos = 0.0, 0
    expertos.modelo.eval()
    for par in pares:
        nombre = elegir(par)
        x = _codificar(expertos.tok, par, expertos.modelo.device)
        with torch.no_grad():
            if nombre is None:
                with expertos.modelo.disable_adapter():
                    perdida = expertos.modelo(**x).loss
            else:
                expertos.modelo.set_adapter(nombre)
                perdida = expertos.modelo(**x).loss
        n = int((x["labels"] != -100).sum())
        total += float(perdida) * n
        cuantos += n
    return math.exp(total / cuantos)


def responder(expertos: Expertos, router, vectorizador, pregunta: str, max_tokens: int = 200) -> tuple[str, str]:
    """Orquestador: el tronco elige el experto y ese adaptador responde (sin recargar el modelo)."""
    import torch

    x, _ = embeddings([pregunta], vectorizador)
    dominio = router.predecir(x)[0]
    expertos.modelo.set_adapter(dominio)
    texto = expertos.tok.apply_chat_template([{"role": "user", "content": pregunta}], tokenize=False,
                                             add_generation_prompt=True)
    ids = expertos.tok(texto, return_tensors="pt").to(expertos.modelo.device)
    with torch.no_grad():
        y = expertos.modelo.generate(**ids, max_new_tokens=max_tokens, do_sample=False)
    return dominio, expertos.tok.decode(y[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


def preguntas_semilla():
    textos, etiquetas = [], []
    for d, qs in SEMILLA.items():
        textos += qs
        etiquetas += [d] * len(qs)
    return textos, etiquetas


def limpiar(texto: str) -> str:
    return re.sub(r"\s+", " ", texto).strip()
