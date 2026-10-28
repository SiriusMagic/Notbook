# Laboratorio neuronal

Experimentos propios con redes neuronales, divididos por tema en 9 cuadernos. Cada sección es un experimento independiente: ábrela y ejecútala de arriba hacia abajo.

> Los resultados guardados (gráficos, tablas) se limpiaron para que los archivos sean livianos. Vuelve a ejecutar una sección para ver sus resultados.

## Cómo abrir un cuaderno

En Colab: **Archivo → Abrir cuaderno → GitHub** → `SiriusMagic/Notbook` → carpeta `laboratorio-neuronal`.

## Cuadernos

| Cuaderno | Tema | Secciones |
|---|---|---|
| [`01_texto_y_lenguaje.ipynb`](./01_texto_y_lenguaje.ipynb) | Extracción de PDF, tokenización, sílabas, agentes de RL sobre texto, BERT, difusión de texto. | 14 |
| [`02_capas_y_activaciones.ipynb`](./02_capas_y_activaciones.ipynb) | Capas con memoria, activaciones adaptativas, neuronas resonantes y polinómicas, memoria híbrida, redes desde cero con NumPy. | 32 |
| [`03_neuronas_alternativas.ipynb`](./03_neuronas_alternativas.ipynb) | Neurona DC, neuronas flotantes, neuronas basadas en k-means, KNN, árbol de decisión, bayes y algoritmos genéticos. | 10 |
| [`04_vision_e_imagenes.ipynb`](./04_vision_e_imagenes.ipynb) | Redes convolucionales auto-decorrelacionadas, reconstrucción con tripletas, GAN, video y un clasificador de imágenes. | 13 |
| [`05_atencion.ipynb`](./05_atencion.ipynb) | Atención multi-cabeza, atención suave con RL, atención local y atención a la salida. | 7 |
| [`06_refuerzo_y_cerebro_modular.ipynb`](./06_refuerzo_y_cerebro_modular.ipynb) | Votación de decisiones, entornos multicuerpo, biomas neuronales, cerebro modular con expertos y tálamo cortical. | 24 |
| [`07_grafos_y_generativos.ipynb`](./07_grafos_y_generativos.ipynb) | Redes de grafos (GCN), fusión condicional y generación multimodal (HCMG), inversión de gradiente. | 12 |
| [`08_optimizacion_y_series.ipynb`](./08_optimizacion_y_series.ipynb) | Tasa de aprendizaje dinámica, optimización por error, detector de sobre-generalización, series sintéticas, clasificador caótico. | 6 |
| [`09_notas_e_ideas.ipynb`](./09_notas_e_ideas.ipynb) | Explicaciones y lluvias de ideas guardadas como texto (modelos autorregresivos, propuestas de arquitecturas). | 10 |
| [`11_texto_contra_industria.ipynb`](./11_texto_contra_industria.ipynb) | Las ideas del cuaderno 01 contra la industria: sílabas vs BPE/GPT-4o, posición jerárquica, segmentador de sílabas y agentes de atención vs TextCNN/Transformer/fastText. | 5 |
| [`10_banco_de_evaluacion.ipynb`](./10_banco_de_evaluacion.ipynb) | Competencia en Colab (GPU): tus piezas del 02 y 04 contra CNN, TextCNN, Transformer, CNN 1D y LSTM, con datos reales de Hugging Face (`laboratorio/banco/`). | 7 |

## Índice por cuaderno

### Texto y lenguaje — `01_texto_y_lenguaje.ipynb`

- Extraer texto de PDF
- Limpieza y tokenización de texto
- Agente PPO para segmentar texto en español
- Segmentador de sílabas
- Tokenizador entrenado con RL
- Difusión de texto
- Clasificador BERT con corrección GNN
- Agente de atención para clasificar texto
- Preguntas y respuestas con BERT
- ordenar.por.longitud
- texto_por_silabas
- DistilBertForQuestionAnswering
- Textus Diffusio Restituo

### Capas y activaciones experimentales — `02_capas_y_activaciones.ipynb`

- Neuronas de ingeniería
- CapaConMemoriaErrores
- Neurona con memoria y atención
- Red con PReLU adaptativa
- Código de Comparativa: Red Estándar vs. Red con Conexiones Denso-Cruzadas
- Red con características densas de entrada
- funciones de activación adaptativas o aprendibles
- funcion adaptativa cambiante
- Propuesta: Activación Basada en la Divergencia de Información (Information Diver
- Neurona polinómica dinámica (clasificador de círculos)
- Neurona resonante
- Neurona con historial (series temporales)
- Aumento de Ruido de Modelo Dinámico (Dynamic Model Noise Augmentation - DMNA)
- Red con Optimización por Mutación Global (GMO)
- Modelo de Destilación de Conceptos (CDM)
- Red con Sesgo de Memoria Residual (RMB)
- Un Modelo de Varianza Experimental: La Red de Caos Controlado (Controlled Chaos 
- Capa Densa con Neuronas de Memoria Híbrida (HMNLayer)
- Red Estándar (Organismo Procariota): Neuronas lineales simples
- inicializador de pesos
- Red desde cero con NumPy
- biolagicamwnte inspirados
- Red desde cero con NumPy (variante)
- NeuronaEstadoInterno
- SimpleOptiNet
- estado interno adaptativo
- SGD con Momentum
- CapaRuteoAdaptativo

### Neuronas alternativas — `03_neuronas_alternativas.ipynb`

- Neurona DC: red ajustable
- Neurona DC: control de hiperparámetros
- Neurona DC: propagación hacia adelante
- Neurona DC: notas
- Neuronas flotantes cargadas desde archivos
- neuros en un arreglo k means
- arreglo knn
- arreglo árbol de decisiones
- bayesiano
- neuronas con algoritmos Genéticos

### Visión e imágenes — `04_vision_e_imagenes.ipynb`

- Red Neuronal con Extracción de Características Orto-Diversas (ODFE-NN)
- Red Neuronal Convolucional Auto-Orto-Diversa (AODC-NN)
- Red Neuronal con Dinámica de Re-extracción Consciente (RDCR-NN)
- Red Neuronal de Auto-Organización con Ruteo de Información Decorrelacionado (NID
- Modelo de Fusión Contextual Recíproca (RCF-NN) para Análisis Secuencial
- Reconstrucción de imágenes con tripletas
- CNN con imágenes marcadas
- Activación seno en imágenes
- GAN: generador y discriminador
- Videopre-processing
- FrameInterpolator
- Prototipos pequeños

### Atención — `05_atencion.ipynb`

- Atención multi-cabeza
- Cabeza de atención suave con RL
- Atención suave con RL sobre imágenes
- AttentiveLinear
- MultiHeadAttention
- Atención local en series sinusoidales
- atwncion a la salida

### Refuerzo y cerebro modular — `06_refuerzo_y_cerebro_modular.ipynb`

- neuronas individuales con RL
- Entorno RL para código Python
- Red de votación de decisiones (RL)
- Auto-Optimización por Sacarina
- 1. Estado s
- Entorno visual multicuerpo
- Empujar cubos con varios cuerpos (RL)
- Ann Biomas
- RL cerebro
- 🧠 Prototipo de RL + Cerebro Modular con MoE
- NeuralBiomesModel
- Modelo Multimodal Bioinspirado: Imagen + Texto Sintético
- Ruteo de salidas de expertos
- clasificación (memoria)
- calculo
- 🧠📡 Diseño: Tálamo Cortical Modular (Router de Biomas)
- 1. Un tronco (corteza central) que decide dónde se procesa cada input
- Router
- ReroutingNetwork
- Agente de direcciones aleatorias
- Agente aleatorio

### Grafos y modelos generativos — `07_grafos_y_generativos.ipynb`

- Normalización adaptativa y fusión condicional (HCMG)
- Red de grafos (GCN)
- nx.draw_networkx_nodes
- GradientReversalFunction
- ReversePredictionNet
- Graphus Nodea Visualis
- vector de condición
- ConditionalFusionBlock (Nexus Condicionalis)
- HCMG (Generatio Multimodalis Hierarchica)
- generadores_condicionales.py)
- (ReversioGradientisFunctio)
- (StratumReversioGradientis)

### Optimización y series temporales — `08_optimizacion_y_series.ipynb`

- Series Temporales Sintéticas
- weighted_vote
- Chaotic20DClassifier
- ProgramadorDeTasaDeAprendizajeDinamico
- OptimizaciónAdaptativaBasadaEnError
- SistemaInmuneCognitivo

### Notas e ideas — `09_notas_e_ideas.ipynb`

- Modelos Autorregresivos
- Modelos Autoregresivos (Autoregressive Models)
- 1. Aprendizaje de Representaciones con Conciencia (Conscious Representation Lear
- 1. Regresión Lineal (Linear Regression)
- diálogo entre expertos, con diversidad de personalidades o enfoques, y un router

## Origen

Reúne los cuadernos de Colab `Bh1`, `Copia de B1` (sus 18 celdas propias), `0.1v` y `Untitled2`. Se quitaron 6 celdas repetidas y los títulos sin sentido se cambiaron por nombres descriptivos.

## Piezas pasadas a código con pruebas

Las partes reutilizables (sílabas, series sintéticas, programador de tasa de aprendizaje, inicializador de Lorenz, voto ponderado e inversión de gradiente) ya están en [`../laboratorio/`](../laboratorio) como archivos `.py` con pruebas, y con varios errores corregidos. Los cuadernos se mantienen como registro de los experimentos.
