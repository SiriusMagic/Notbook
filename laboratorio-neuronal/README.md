# Laboratorio neuronal

Experimentos propios con redes neuronales: neuronas con memoria, activaciones adaptativas, aprendizaje por refuerzo, redes de grafos, atención, modelos "bioinspirados" y más.

Todo está en un solo cuaderno, [`laboratorio_neuronal.ipynb`](./laboratorio_neuronal.ipynb), que reúne:

| Cuaderno original | Qué se tomó |
|---|---|
| `Bh1` | Todo (era la versión más completa). Se quitaron 6 celdas repetidas. |
| `Copia de Bh1` | Nada: era idéntico a `Bh1`. |
| `Copia de Copia de Bh1` | Nada: era una versión anterior; todo su código ya estaba en `Bh1`. |
| `Copia de B1` | Las 18 celdas que no estaban en `Bh1` (sección **Aportes de B1**). |
| `0.1v` | Clasificador de imágenes (sección **Prototipos pequeños**). |
| `Untitled2` | Agente aleatorio (sección **Prototipos pequeños**). |

> Los resultados guardados (gráficos, tablas) se limpiaron: los cuatro cuadernos de origen sumaban ~27 MB y este pesa ~2 MB. Vuelve a ejecutar una sección para ver sus resultados.
>
> Las secciones con títulos sin sentido se renombraron según lo que contienen; el nombre original aparece entre paréntesis, por ejemplo `## Red de grafos (GCN) (antes: «kdksk»)`.

## Cómo abrirlo

En Colab: **Archivo → Abrir cuaderno → GitHub**, escribe `SiriusMagic/Notbook` y elige `laboratorio-neuronal/laboratorio_neuronal.ipynb`.

No lo ejecutes entero de una vez: cada sección es un experimento independiente con sus propias instalaciones (`pip install`). Ejecuta solo la que te interese, de arriba hacia abajo.

## Índice por temas

Los números son la posición aproximada de la sección dentro del cuaderno.

### Texto y lenguaje
- Extraer texto de PDF · Limpieza y tokenización · Segmentador de sílabas
- Agente PPO para segmentar texto · Tokenizador entrenado con RL
- Clasificador BERT con corrección GNN · Preguntas y respuestas con BERT
- Difusión de texto · Textus Diffusio Restituo (B1) · Tokenizador por sílabas con embeddings jerárquicos

### Neuronas y capas experimentales
- Capa con memoria de errores · Neurona con memoria y atención · Neurona de estado interno
- PReLU adaptativa · Integración no lineal · Conexiones denso-cruzadas
- Activaciones adaptativas y aprendibles · Activación por divergencia (IDA)
- Neurona DC (varias versiones) · Neurona resonante · Neurona polinómica dinámica · Neuronas flotantes
- Neurona dendrítica · Synaptic tagging · Memoria híbrida (HMN) · Sesgo de memoria residual (RMB)
- Ruido dinámico (DMNA) · Mutación global (GMO) · Destilación de conceptos (CDM) · Red caótica (CCN)
- Neuronas como k-means, KNN, árbol de decisión, bayesiano y algoritmos genéticos
- Inicializador de pesos con ruido Perlin y atractor de Lorenz

### Visión e imágenes
- Redes convolucionales auto-decorrelacionadas (AODC, RDCR, NIDI, ODFE, RCF-NN)
- Reconstrucción de imágenes con tripletas · CNN con imágenes marcadas · Activación seno
- GAN (generador y discriminador) · Interpolación de cuadros de video · Preprocesado de video
- Clasificador de imágenes (antes `0.1v`)

### Atención
- Atención multi-cabeza · Atención suave con RL (texto e imágenes) · Atención local en series
- Atención a la salida (AttentiveLinear)

### Aprendizaje por refuerzo y "cerebro modular"
- Red de votación de decisiones · Neuronas individuales con RL · Entorno RL para código Python
- Entornos multicuerpo (empujar cubos) con Stable-Baselines3
- Biomas neuronales (NeuralBiomesModel) · Cerebro modular con expertos (MoE, router, memoria episódica)
- Tálamo cortical modular · Red de re-ruteo · Agente aleatorio (antes `Untitled2`)

### Grafos y generativos
- Red de grafos (GCN) · Visualización de grafos (B1)
- Normalización adaptativa, fusión condicional y generación multimodal jerárquica (HCMG)
- Inversión de gradiente (GradientReversal / Reversio Gradientis)

### Optimización y datos
- Programador dinámico de tasa de aprendizaje · Optimización adaptativa por error (B1)
- Sistema inmune cognitivo (detector de sobre-generalización)
- Series temporales sintéticas · Clasificador caótico 20D · Regresión y modelos autorregresivos (notas)

## Siguiente paso sugerido

Cuando elijas los experimentos que valen la pena, se pueden pasar a archivos `.py` con pruebas, como se hizo con `atlas-rf`.
