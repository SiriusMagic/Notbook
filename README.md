# Notbook

Laboratorio de investigación en redes neuronales: tus neuronas, capas y arquitecturas propias, medidas contra la tecnología que usa hoy la industria. Cuadernos de Google Colab de varios años, ordenados y documentados.

## ¿Qué funciona? Resultados en una mirada

Cada pieza se midió en el **banco de evaluación** (`laboratorio/banco/`), contra la red estándar y contra lo que usa la industria, con varias semillas y datos de prueba que la red nunca vio.

- 🟢 **prometedora:** le gana o casi le gana a la industria en algo concreto.
- ⚪ **neutra:** empata; no daña, pero tampoco aporta.
- 🔴 **no está lista:** pierde por un error de diseño (anotado y, cuando se pudo, arreglado).

### 🟢 Las más prometedoras

| # | Pieza (cuaderno) | Campo | Resultado | Contra la industria |
|---|---|---|---|---|
| 1 | **Tripletas con flujo** (04, mejoradas) | Video: inventar el fotograma del medio | 19,22 dB con 24 mil parámetros; con 4 fotogramas, 20,25 dB | U-Net: 19,13 dB con 467 mil (con 4 fotogramas, 19,79). **Empatan o ganan por poco con 19 veces menos parámetros** |
| 2 | **Stacking** (recuperado de tu cuaderno completo) | Combinar modelos | 98,5 % en dígitos (lo mejor medido); 83,6 % en la tarea caótica | Supera a la votación ponderada en 4 de 5 tareas; Gradient Boosting sigue mejor en tablas (94 %) |
| 3 | **Neurona dendrítica** (02) | Datos en tabla con ruido | 76,1 % en la tarea caótica (red estándar: 70,0 %) | Casi iguala a Dropout (77,4 %) con los mismos parámetros |
| 3b | **Base radial entrenable** (03, reparada: antes neuronas KNN) | Datos en tabla con ruido | 76,9 % en la tarea caótica (red estándar: 70,5 %) | Casi iguala a Dropout (77,4 %); antes perdía en las 7 tareas |
| 4 | **Corrector GNN con residuales** (01) | Corrección ortográfica | Baja las letras mal de 12,6 % a 10,8 % | El diccionario clásico (Norvig/SymSpell) llega a 6,7 %; falta medir en GPU contra BiLSTM y Transformer |
| 5 | **Tu difusión con máscaras** (01, recuperada) | Rellenar huecos en un texto | 3,0 % de acierto al rellenar | Empata con la difusión de la industria (MDLM: 3,4 %); al escribir de corrido gana un GPT pequeño |
| 6 | **DMNA** y **genética invertida** (02, 03) | Datos en tabla con ruido | 72,2 % en la tarea caótica | Ayudan poco frente a la red estándar (70,0 %); Dropout da 77,4 % |

### Todos los resultados en una tabla

Lo mejor de cada familia contra la industria. 🟢 gana o empata usando mucho menos · ⚪ empata · 🔴 pierde. El detalle de cada fila está en su informe.

| Familia | Lo tuyo (mejor versión) | Industria | Veredicto | Informe |
|---|---|---|---|---|
| **Tokenizadores** (clasificar reseñas, CNN) | Sílabas: 39,9 % (2,0 tokens por palabra) | BPE: 35,2 % · Unigram: 36,7 % · GPT-4o: 34,7 % | ⚪ un poco arriba, sin diferencia significativa | [01-tokenizadores…](./laboratorio/banco/informes/01-tokenizadores-segmentadores-corrector.md) |
| **Segmentador de sílabas** | BiLSTM: 99,2 % de palabras exactas | pyphen (LibreOffice, TeX): 71,1 % | 🟢 | [01-tokenizadores…](./laboratorio/banco/informes/01-tokenizadores-segmentadores-corrector.md) |
| **Corrector ortográfico** | GNN con residuales: 10,8 % de letras mal | Diccionario: 6,7 % (sin corregir: 12,6 %) | 🔴 mejora el texto, pero el diccionario gana | [01-tokenizadores…](./laboratorio/banco/informes/01-tokenizadores-segmentadores-corrector.md) |
| **Generar texto** | Difusión con máscaras: rellena huecos 3,0 % | MDLM: 3,4 % · GPT pequeño: perplejidad 263 contra 803 | ⚪ rellenar · 🔴 escribir de corrido | [01-generativos…](./laboratorio/banco/informes/01-generativos-y-refuerzo.md) |
| **Predecir la siguiente palabra** | Agente PPO: 9-12 % | Red supervisada: 16,7 % | 🔴 | [01-generativos…](./laboratorio/banco/informes/01-generativos-y-refuerzo.md) |
| **Capas y activaciones** (unas 60, 7 tareas) | Dendrítica: 76,1 % en la caótica · base radial: 76,9 % | GELU, Dropout (77,4 %), BatchNorm, LayerNorm | ⚪ casi todas empatan; 🟢 las dos de la izquierda en tablas con ruido | [02-…](./laboratorio/banco/informes/02-dendritica-y-otras.md), [reparadas](./laboratorio/banco/informes/reparadas.md) |
| **Piezas reparadas** (8 «no listas») | Las 8 mejoran; 5 ya no pierden en nada | — | ⚪ / 🟢 base radial | [reparadas](./laboratorio/banco/informes/reparadas.md) |
| **Modelos clásicos** (tablas) | Stacking: 98,5 % en dígitos | Gradient Boosting: 94 % en la caótica | 🟢 en dígitos · 🔴 en tablas | [03-clasicos](./laboratorio/banco/informes/03-clasicos.md), [08-…](./laboratorio/banco/informes/08-votacion-stacking-y-caotica.md) |
| **Video: fotograma del medio** | Tripletas con flujo: 19,22 dB, 24 mil parámetros | U-Net: 19,13 dB, 467 mil | 🟢 empata con 19 veces menos | [04-video](./laboratorio/banco/informes/04-video.md) |
| **Video: desde 4 fotogramas** | Tripletas con flujo: 20,25 dB, 27 mil | U-Net: 19,79 dB, 469 mil (FrameInterpolator original: 17,15 dB con 66 millones) | 🟢 | [04-video](./laboratorio/banco/informes/04-video.md) |
| **Atención** | Transformer con máscara · atención dinámica (reparada) | Transformer | ⚪ | [05-atencion](./laboratorio/banco/informes/05-atencion.md) |
| **Refuerzo** | Cubos con guía: 65 % · REINFORCE estándar: 35,6 de 35,8 | PPO con guía (*reward shaping*) | 🟢 con guía · 🔴 escribir código | [06-refuerzo](./laboratorio/banco/informes/06-refuerzo.md) |
| **Generativos** | D-ReG: perplejidad 176 · HCMG: 19,29 dB | GRU del mismo tamaño: 146 · U-Net: 19,80 dB | 🔴 | [07-dreg-y-hcmg](./laboratorio/banco/informes/07-dreg-y-hcmg.md) |
| **Adaptar a datos nuevos** | Anti-gradiente como DANN: 42,2 % | Red normal: 39,1 % | ⚪ (+3 puntos, no significativo) | [reparadas](./laboratorio/banco/informes/reparadas.md) |
| **Router + mini-expertos LoRA** | Cuaderno 14 | Un solo LoRA para todo | ⏳ falta correr en GPU | [cuaderno 14](./laboratorio-neuronal/14_router_y_expertos_lora.ipynb) |

### Lo que se aprendió

- **En tareas pequeñas casi nada propio gana**: de unas 60 piezas, la gran mayoría empata con la red estándar. Las «victorias» de los cuadernos originales no se sostenían con varias semillas y datos de prueba.
- **Más parámetros no es mejor**: en la tarea caótica, una red estándar de 8 neuronas (75,6 %) le gana a una de 32 (70,0 %), porque la grande memoriza. Varias «victorias» (cerebro modular, biomas, integración no lineal, D-ReG) se explican por el tamaño. Una pieza vale la pena si le gana a la red estándar **de su mejor tamaño**.
- **Cuidado con la fuga de datos**: HCMG parecía ganarle a la U-Net (20,44 contra 20,04 dB) hasta que se vio que los dígitos de prueba se repetían con los de entrenamiento. Con dígitos nunca vistos, pierde (19,29 contra 19,80). Por eso cada resultado se mide con datos que la red no vio.
- **Los errores de diseño más comunes**: atención sobre un solo elemento (no elige nada), LSTM con secuencias de largo 1 (no recuerda nada), recompensas al azar o que llegan demasiado tarde, y objetivos imposibles.

### Dónde ver cada cosa

| Quiero… | Dónde |
|---|---|
| Saber **qué pieza usar en cada campo** y cuáles no están listas (con el motivo) | [`RECOMENDACIONES.md`](./RECOMENDACIONES.md) |
| Entender **cómo funciona cada pieza** y probarla yo mismo (corre en CPU) | [`laboratorio-neuronal/12_como_funciona_cada_pieza.ipynb`](./laboratorio-neuronal/12_como_funciona_cada_pieza.ipynb) |
| Ver **las pruebas y las cifras** de cada cuaderno | [`laboratorio/banco/informes/`](./laboratorio/banco/informes): un informe por tema, con un índice |
| Saber **cómo se mide** (semillas, veredicto, tareas) y repetir las pruebas | [`laboratorio/banco/README.md`](./laboratorio/banco/README.md) |
| Correr **en GPU** lo que en CPU quedó corto (un solo cuaderno, deja todo en tu Drive) | [`laboratorio-neuronal/13_correr_pendientes_en_gpu.ipynb`](./laboratorio-neuronal/13_correr_pendientes_en_gpu.ipynb) |
| Correr las competencias **con GPU y datos reales** por separado | cuadernos [`10`](./laboratorio-neuronal/10_banco_de_evaluacion.ipynb) y [`11`](./laboratorio-neuronal/11_texto_contra_industria.ipynb) en Colab |
| Ver **todos los cuadernos** y sus secciones | [`laboratorio-neuronal/README.md`](./laboratorio-neuronal/README.md) |

### Cómo se probó

- **Mismas tareas para todos:** regresión, círculos, lunas, XOR, serie de seno, dígitos, una tarea caótica de 20 dimensiones, reseñas reales en español, videos de dígitos en movimiento y entornos de refuerzo.
- **Varias semillas** (3 a 5): si una pieza gana con una y pierde con otra, fue suerte.
- **Datos de prueba** que la red no vio al entrenar.
- **Veredicto estadístico:** prueba t pareada, corregida por comparaciones múltiples.
- **Rivales de la industria** en cada campo: GELU, Dropout, BatchNorm, CNN, U-Net, LSTM, GRU, Transformer, BPE, MDLM, Gradient Boosting, PPO…
- Para repetirlas: `python -m laboratorio.banco --todo` (tareas pequeñas, unos 5 minutos en CPU) y los demás comandos en [`CLAUDE.md`](./CLAUDE.md).

## Contenido

| Carpeta | Qué es |
|---|---|
| [`laboratorio-neuronal/`](./laboratorio-neuronal) | Los cuadernos: 01 texto, 02 capas y activaciones, 03 neuronas alternativas, 04 visión y video, 05 atención, 06 refuerzo y cerebro modular, 07 grafos y generativos, 08 optimización y series, 09 notas e ideas; 10 y 11 competencias con GPU; 12 guía de cómo funciona cada pieza; 13 correr lo pendiente en GPU; 14 router con mini-expertos LoRA. |
| [`laboratorio/`](./laboratorio) | Las piezas pasadas a código `.py` con pruebas, y el **banco de evaluación** (`laboratorio/banco/`) con sus informes (`laboratorio/banco/informes/`). |
| [`vulkan/`](./vulkan) | Portal de acceso de la red Wi-Fi Vulkan (`index.html`). |

## Seguridad y mantenimiento

- **Revisión automática** (`.github/workflows/check.yml`): en cada cambio comprueba que los cuadernos tengan formato válido, que no tengan llaves de API ni resultados guardados, y avisa de errores de sintaxis. También corre las pruebas de `laboratorio/` (`pytest`).
- **Dependabot** vigila `laboratorio-neuronal/requirements.txt` y `laboratorio/requirements.txt`: avisa de fallos de seguridad y propone actualizaciones una vez al mes, agrupadas.
- `torch.load` usa `weights_only=True` (solo carga pesos, no código) y los archivos `joblib` llevan un aviso de cargar solo archivos propios.
- PyTorch Geometric se instala desde PyPI, sin instalaciones desde URLs externas.

<details>
<summary><b>Qué pasó con cada cuaderno original</b></summary>

| Cuaderno | Destino |
|---|---|
| `Bh1` | Base de `laboratorio-neuronal`, dividido en 9 cuadernos por tema. |
| `Copia de B1` | Sus 18 celdas propias se añadieron a `laboratorio-neuronal`. |
| Cuaderno completo de Drive (451 celdas, con llaves de API) | Era la versión completa de `Bh1`, ya repartida en los cuadernos 01-09. Lo único nuevo, el **sistema inmune cognitivo**, pasó al cuaderno 09. Las utilidades de Python sueltas (PDF, `async`, `__slots__`, primos) y la demo de chat con un LLM local se descartaron. Las llaves **no** se subieron. |
| `0.1v`, `Untitled2` | Añadidos a `laboratorio-neuronal` como prototipos pequeños. |
| `Copia de vulkan` | Convertido en `vulkan/index.html`. |
| `Copia de Antenas`, `Copia de Untitled10` | Movidos a **atlas-rf** → `experiments/`. |
| `Copia de Untitled4`, `Copia de Untitled9` | Movidos a **lica-player** → `herramientas/`. |
| `Copia de Bh1`, `Copia de Copia de Bh1` | Borrados: duplicados de `Bh1`. |
| `Untitled0` | Borrado: descargador con `pytube`, reemplazado por el de lica-player. |
| `Untitled3`, `Copia de Untitled5`, `Copia de Untitled8` | Borrados: pruebas sin valor. |
| `Copia de Untitled1` | Borrado: contenía datos personales. |
| `Copia de Untitled0/2/3/6/7`, `Untitled1`, `Untitled4`, `Copia de S.C.I.M.` | Borrados: vacíos. |

Todo lo borrado sigue recuperable en el historial de Git.

</details>

## Abrir en Colab

**Archivo → Abrir cuaderno → GitHub** → escribe `SiriusMagic/Notbook` y elige el cuaderno. Como el repo es privado, Colab te pedirá autorizar el acceso a GitHub la primera vez.

## Licencia

Copyright 2026 SiriusMagic. Apache 2.0: consulta [`LICENSE`](./LICENSE).
