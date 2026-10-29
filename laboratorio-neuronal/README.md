# Laboratorio neuronal

Experimentos propios con redes neuronales: 9 cuadernos por tema (01-09), 2 cuadernos para competir en GPU (10-11), una guía de cómo funciona cada pieza (12) uno para correr en GPU lo pendiente (13) y el router con mini-expertos LoRA (14). Cada sección es un experimento independiente: ábrela y ejecútala de arriba hacia abajo.

> Los cuadernos se guardan sin resultados (gráficos, tablas) para que sean livianos. Vuelve a ejecutar una sección para ver sus resultados. Las cifras medidas están en los informes del banco ([`../laboratorio/banco/informes/`](../laboratorio/banco/informes)) y el resumen de qué funciona, en el [README principal](../README.md).

## Cómo abrir un cuaderno

En Colab: **Archivo → Abrir cuaderno → GitHub** → `SiriusMagic/Notbook` → carpeta `laboratorio-neuronal`.

## Cuadernos

| Cuaderno | Tema | Qué tiene | Resultados |
|---|---|---|---|
| [`01_texto_y_lenguaje.ipynb`](./01_texto_y_lenguaje.ipynb) | Texto y lenguaje | Sílabas y tokenización, agente PPO que predice palabras, tokenizador con RL, difusión de texto, corrector GNN y agentes de atención. | [informe](../laboratorio/banco/informes/01-generativos-y-refuerzo.md) |
| [`02_capas_y_activaciones.ipynb`](./02_capas_y_activaciones.ipynb) | Capas y activaciones | Capas con memoria, activaciones adaptativas, neuronas resonantes, polinómicas y dendríticas, DMNA, GMO, CCN, inicializadores Perlin y Lorenz. | [informe](../laboratorio/banco/informes/02-primera-tanda.md) |
| [`03_neuronas_alternativas.ipynb`](./03_neuronas_alternativas.ipynb) | Neuronas alternativas | Neuronas flotantes y en el espacio, árbol fonológico, neuronas KNN, de árbol y bayesianas, redes genéticas y clasificadores clásicos. | [informe](../laboratorio/banco/informes/03-neuronas-alternativas.md) |
| [`04_vision_e_imagenes.ipynb`](./04_vision_e_imagenes.ipynb) | Visión y video | CNN auto-decorrelada, «papelito preguntón», RCF, tripletas e interpolación de video, GAN y clasificador de imágenes. | [informe](../laboratorio/banco/informes/04-video.md) |
| [`05_atencion.ipynb`](./05_atencion.ipynb) | Atención | Atención multi-cabeza, atención suave con RL (texto e imágenes), atención a la salida y atención local. | [informe](../laboratorio/banco/informes/05-atencion.md) |
| [`06_refuerzo_y_cerebro_modular.ipynb`](./06_refuerzo_y_cerebro_modular.ipynb) | Refuerzo y cerebro modular | Neuronas con RL, entornos (escribir código, empujar cubos), biomas, cerebro modular con expertos, tálamo y ReroutingNetwork. | [informe](../laboratorio/banco/informes/06-refuerzo.md) |
| [`07_grafos_y_generativos.ipynb`](./07_grafos_y_generativos.ipynb) | Grafos y generativos | HCMG, GCN y GAT, neuronas espaciales, anti-gradiente, dibujos de grafos y D-ReG. | [informe](../laboratorio/banco/informes/07-dreg-y-hcmg.md) |
| [`08_optimizacion_y_series.ipynb`](./08_optimizacion_y_series.ipynb) | Optimización y series | Votación ponderada, stacking y la tarea caótica de 20 dimensiones. | [informe](../laboratorio/banco/informes/08-votacion-stacking-y-caotica.md) |
| [`09_notas_e_ideas.ipynb`](./09_notas_e_ideas.ipynb) | Notas e ideas | Explicaciones e ideas guardadas como texto: modelos autorregresivos, propuestas de arquitecturas, cadena de pensamiento y sistema inmune cognitivo. | — |
| [`10_banco_de_evaluacion.ipynb`](./10_banco_de_evaluacion.ipynb) | Competencia con GPU | Tus piezas contra CNN, TextCNN, Transformer, CNN 1D y LSTM con datos reales de Hugging Face. | — |
| [`11_texto_contra_industria.ipynb`](./11_texto_contra_industria.ipynb) | Texto contra la industria (GPU) | Sílabas contra BPE y GPT-4o, posición jerárquica, segmentadores, corrector GNN y agentes de atención contra TextCNN, Transformer y fastText. | — |
| [`12_como_funciona_cada_pieza.ipynb`](./12_como_funciona_cada_pieza.ipynb) | Guía: cómo funciona cada pieza | Cómo funciona cada neurona, capa y arquitectura, qué resultado dio y por qué el tamaño cambia el resultado. Corre en CPU. | — |
| [`13_correr_pendientes_en_gpu.ipynb`](./13_correr_pendientes_en_gpu.ipynb) | Correr lo pendiente en GPU | Repite con más datos las mediciones que en CPU quedaron cortas y deja los informes en un `.zip` en tu Drive. | — |
| [`14_router_y_expertos_lora.ipynb`](./14_router_y_expertos_lora.ipynb) | Router troncal + mini-expertos LoRA | Un clasificador pequeño elige qué adaptador LoRA de oficio usar (electricidad, agricultura, mecánica, redes). Las etiquetas las pone DeepSeek si guardas `DEEPSEEK_API_KEY` en los Secretos de Colab; si no, un modelo pequeño de Hugging Face (Qwen 1,5B); si tampoco, reglas. Necesita GPU. | ⏳ |

## Índice de secciones (cuadernos 01-09)

### 01 · Texto y lenguaje

- Agente PPO que predice la siguiente palabra
- Segmentador de sílabas
- Tokenizador entrenado con RL
- Difusión de texto
- Clasificador BERT con corrección GNN
- Agente de atención para clasificar texto

### 02 · Capas y activaciones

- Neuronas de ingeniería
- Memoria de errores, memoria de gradiente, umbral dinámico y LR por neurona
- Neurona con memoria y atención
- Red con PReLU adaptativa
- Neurona de integración no lineal
- Red con conexiones denso-cruzadas
- Red con características densas de entrada
- Activación adaptativa (mezcla de ReLU y sigmoide)
- Activaciones IDA (divergencia de información) y SMA (activación que muta)
- Neurona polinómica dinámica (clasificador de círculos)
- Neurona resonante
- Aumento de ruido dinámico (DMNA)
- Optimización por mutación global (GMO)
- Destilación de conceptos (CDM)
- Sesgo con memoria residual (RMB)
- Red de caos controlado (CCN)
- Capa con neuronas de memoria híbrida (HMN)
- HMN contra LSTM
- Inicializador de pesos (ruido Perlin y atractor de Lorenz)
- CNN con entrada densa (MNIST)
- Neurona dendrítica (NumPy)
- Synaptic tagging y estado interno
- Ruteo adaptativo de contribución (RARC)

### 03 · Neuronas alternativas

- Neuronas flotantes
- Red semántica (palabras → conceptos)
- Árbol fonológico (letras → sílabas → palabras)
- Neuronas en un espacio 3D (conexiones por distancia)
- Neurona con arreglo KNN
- Neurona con árbol de decisión
- Clasificadores clásicos (la referencia de la industria para datos en tabla)
- Neurona bayesiana (Naive Bayes)
- Redes genéticas

### 04 · Visión y video

- Red convolucional auto-decorrelada (AODC-NN)
- El «papelito preguntón»: extracción de características orto-diversas (ODFE-NN)
- Fusión contextual recíproca recurrente (RCF-NN)
- Reconstrucción de imágenes con tripletas
- FrameInterpolator: el fotograma del medio a partir de 4
- CNN con imágenes marcadas
- Activación seno en imágenes
- GAN: generador y discriminador
- Preparar el video: fotogramas en `.npy`
- FrameInterpolator: predecir el fotograma siguiente

### 05 · Atención

- Atención multi-cabeza (la estándar, como referencia)
- Atención suave con RL (texto)
- Atención suave con RL sobre imágenes
- Atención a la salida (AttentiveLinear)
- Atención local en series sinusoidales

### 06 · Refuerzo y cerebro modular

- Neuronas individuales con RL (RlNeuron)
- Entorno RL para código Python
- Red de votación de decisiones (RL)
- Auto-optimización por sacarina (notas)
- Entorno visual multicuerpo
- Empujar cubos con varios cuerpos (RL)
- Biomas neuronales (NeuralBiomesModel)
- Cerebro modular: prototipo con memoria episódica (CognitiveMoE)
- Cerebro modular multimodal: imagen + texto sintético (CorticalMoE)
- Tálamo con 4 biomas y retroalimentación (ThalamusBiomeNet)
- Router cerebral: mezcla suave de expertos (CerebralRouterNet)
- Cerebro modular final con MNIST (CorticalMoE)
- ReroutingNetwork (REINFORCE en un problema pequeño)

### 07 · Grafos y generativos

- Normalización adaptativa y fusión condicional (HCMG)
- Redes de grafos de la industria (GCN y GAT con Cora)
- Neuronas espaciales
- Capa anti-gradiente (inversión de gradiente)
- Visualización de grafos de nodos
- Dibujos de grafos ramificados (networkx)
- Generador de texto recurrente dual (D-ReG)

### 08 · Optimización y series

- Votación ponderada (weighted_vote)
- Tarea caótica 20D (Chaotic20DClassifier)

### 09 · Notas e ideas

- Cadena de pensamiento (traído del cuaderno 01)
- Sistema inmune cognitivo

## Piezas pasadas a código con pruebas

Las ideas medidas están como código `.py` con pruebas en [`../laboratorio/banco/`](../laboratorio/banco), y las piezas reutilizables (sílabas, series sintéticas, programador de tasa de aprendizaje, inicializadores, voto ponderado, inversión de gradiente) en [`../laboratorio/`](../laboratorio). Los cuadernos se mantienen como registro de los experimentos.

## Origen y limpieza

Reúne los cuadernos de Colab `Bh1`, `Copia de B1`, `0.1v`, `Untitled2` y tu cuaderno completo de Drive. En octubre de 2026 se quitaron las versiones repetidas de cada idea (se dejó la más completa), las celdas que no eran investigación (utilidades sueltas, tutoriales de modelos ya hechos) y se pusieron títulos descriptivos. Todo lo quitado sigue en el historial de Git.
