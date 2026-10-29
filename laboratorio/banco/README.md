# Banco de evaluación de neuronas, activaciones y capas

El cuaderno 02 tiene decenas de experimentos «mi idea contra RedEstandar». Cada uno se probó **una vez**, con una sola semilla y midiendo sobre los mismos datos de entrenamiento. Así no se puede saber si una idea es mejor o si tuvo suerte.

Las recomendaciones por campo (dónde conviene usar cada pieza y cuáles no están listas) están en [`RECOMENDACIONES.md`](../../RECOMENDACIONES.md).

Este banco las mide a todas igual:

- **Mismas tareas**: las que ya usan los cuadernos, con datos fijos y una parte de **prueba** que la red nunca ve al entrenar.
- **Misma red**: dos capas ocultas de 32 neuronas y una salida lineal. Solo cambia la pieza.
- **Varias semillas** (5 por defecto). La pieza y la estándar usan las mismas semillas, así se comparan por pares.
- **Veredicto honesto** con una prueba t pareada (y, al comparar muchas a la vez, corregida con Benjamini-Hochberg para no contar ganadores por azar; con una sola semilla nunca se declara ganador):
  - 🟢 **mejor**: es mejor y la diferencia es real (p < 0,05).
  - ⚪ **igual**: la diferencia cabe dentro del ruido.
  - 🔴 **peor**: es peor, y la diferencia también es real.
  - 💥 **falla**: la red dio NaN («explotó»).

El diseño copia el de [openai/evals](https://github.com/openai/evals): un **registro** de evaluaciones con nombre (`EVALUACIONES`), una función que corre cada evaluación, un **registrador** que guarda cada corrida en JSONL y un **informe**. No se instala `evals`, porque está hecho para evaluar modelos de lenguaje y no redes de PyTorch.

## Uso

```bash
pip install -r laboratorio/requirements.txt            # + torch
python -m laboratorio.banco --todo --semillas 5         # todo (~5 min en CPU)
python -m laboratorio.banco --pieza ida --tarea lunas   # una pieza en una tarea
```

Genera `resultados/informe.md` (una tabla) y `resultados/AAAA-MM-DD.jsonl` (cada corrida). En Colab: cuaderno `laboratorio-neuronal/10_banco_de_evaluacion.ipynb`.

## Tareas

| Tarea | Qué es | Métrica |
|---|---|---|
| `regresion_cubica` | y = x³ − x + ruido; es la más usada en el cuaderno 02 | MSE (menos es mejor) |
| `circulos` | dos círculos concéntricos con ruido | exactitud |
| `lunas` | dos medias lunas | exactitud |
| `xor` | XOR en los 4 cuadrantes | exactitud |
| `serie_seno` | predecir el siguiente valor de una onda con ruido | MSE |
| `digitos` | dígitos 8×8 de scikit-learn (un MNIST pequeño, sin descargas) | exactitud |
| `caotica_20d` | 4 clases por reglas enredadas en 20 dimensiones, con ruido (cuaderno 08) | exactitud |

## Cuaderno 01: texto contra la industria

```bash
python -m laboratorio.banco --tokenizadores --tarea resenas_es   # tokenizadores.md
python -m laboratorio.banco --segmentadores                      # segmentadores.md
python -m laboratorio.banco --correccion                         # correccion.md
python -m laboratorio.banco --competencia --tipo texto           # agentes de atención y piezas vs TextCNN, Transformer, fastText
```

- **Tokenizadores** (`tokenizadores.py`): sílabas por reglas y la heurística original del cuaderno contra palabras, caracteres, BPE (GPT, Llama), SentencePiece Unigram (T5, Gemma) y el de GPT-4o (`tiktoken`).
  - Todos usan el mismo vocabulario: 8 000 unidades.
  - Mide tokens por palabra, desconocidas, textos truncados, velocidad y exactitud con TextCNN y Transformer.
  - Con las sílabas también compite tu **posición jerárquica**, en sus dos versiones: por pares de sílabas (la del cuaderno) y por palabra + sílaba. Se compara contra la posición aprendida y la senoidal.
- **Segmentadores** (`segmentadores.py`): reglas, heurística del cuaderno, pyphen, la BiLSTM de la celda 16 y un Transformer de caracteres.
  - Se miden con 128 palabras silabeadas a mano según la RAE (`silabas_referencia.py`).
- **Corrección ortográfica** (`correccion.py`, celda 26): tu GNN, en tres versiones, contra BiLSTM y Transformer de caracteres y un corrector por diccionario (Norvig/SymSpell).
  - Las tres versiones de tu GNN son: tal cual con refuerzo, tal cual supervisada, y con conexiones residuales.
  - Usa frases reales con los errores de tu propia función, en dos escenarios: solo letras cambiadas, y todos los errores.
  - Se quitó el «puntaje BERT» de la recompensa, porque su capa final nunca se entrenó.
  - Hallazgo: sin conexiones residuales la red olvida la letra original y deja el texto peor que sin corregir. Con ellas mejora el texto con muy pocos parámetros.
- **Agentes de atención** (`arquitecturas.py`):
  - tu atención binaria con RL (celda 31);
  - tu atención por niveles con RL (celdas 32 y 36);
  - tu atención «hebbiana» (celda 33).

  Compiten contra la mejor arquitectura normal y se mide qué parte del texto leen. En el cuaderno, el clasificador y los embeddings no se entrenaban; aquí sí, y REINFORCE usa línea base.

## Mediciones de hardware y por dentro (todas las competencias)

`medicion.py` añade a cada corrida:
- **Hardware**: latencia para un ejemplo (mediana y p95), ejemplos/s y tokens/s, peso de la red y VRAM pico en GPU.
- **Por dentro**: neuronas muertas e inmóviles, normas de activación, gradiente por capa (¿se desvanece?) y eficiencia (exactitud por millón de parámetros).
- **CKA** contra la red normal, junto al CKA de la red normal consigo misma con otra semilla, que es lo que cambia solo por azar.

El informe de la competencia trae una tabla «Hardware y por dentro» por tarea.

## Competencia contra la tecnología normal

```bash
python -m laboratorio.banco --competencia --rapido                     # una tarea por tipo (≈ 15–30 min con GPU)
python -m laboratorio.banco --competencia --tipo imagen --guardar modelos
python -m laboratorio.banco --competencia --tipo texto --pieza ida --pieza seno
```

Cada pieza compite contra la arquitectura que se usa hoy para cada tipo de dato:

| Datos | Arquitecturas | Cómo entra tu pieza |
|---|---|---|
| imagen | `densa`, **`cnn`** (2 bloques conv + BatchNorm) | en la parte final; si es activación (PReLU, IDA, adaptativa, seno), también en lugar de ReLU dentro de la CNN |
| texto | `densa` (TF-IDF), **`texto_cnn`** (TextCNN), **`transformer`** (2 capas, 4 cabezas), **`fasttext`** | en la parte final; en TextCNN también como activación |
| serie | `densa`, **`serie_cnn`** (CNN 1D), **`lstm`** | en la parte final; en la CNN 1D también como activación |

El informe `competencia.md` tiene una **tabla de posiciones** por tarea. Para cada modelo muestra:
- cuánto mejor o peor sale frente a la mejor arquitectura normal;
- si la pieza **mejora la arquitectura donde va**, es decir, la misma arquitectura con ReLU. Usa prueba t pareada y corrección de Benjamini-Hochberg.

Al final viene un **mapa por tipo de dato**: cada pieza contra imágenes, texto y series, con cuántas veces mejora (🟢), iguala (⚪) o empeora (🔴) su arquitectura y una etiqueta (**prometedora**, neutral o «no sirve aquí»). Así se ve de un vistazo si una pieza sirve para imágenes aunque no para texto.

En las series se muestra también la **predicción ingenua** (repetir el último valor). Un modelo que no la supera no sirve para esa serie.

Con `--arquitectura cnn` (sin `--competencia`) se corre el banco normal dentro de una sola arquitectura. La GPU se usa sola si hay (`--dispositivo cpu` para evitarlo).

## Tareas con datos reales (Hugging Face)

| Tarea | Datos | Tipo |
|---|---|---|
| `mnist` | [ylecun/mnist](https://huggingface.co/datasets/ylecun/mnist): dígitos 28×28 | imagen, 10 clases |
| `fashion_mnist` | [zalando-datasets/fashion_mnist](https://huggingface.co/datasets/zalando-datasets/fashion_mnist): ropa 28×28 | imagen, 10 clases |
| `cifar10` | [uoft-cs/cifar10](https://huggingface.co/datasets/uoft-cs/cifar10): fotos a color 32×32 | imagen, 10 clases |
| `resenas_es` | [SetFit/amazon_reviews_multi_es](https://huggingface.co/datasets/SetFit/amazon_reviews_multi_es): reseñas en español | texto, 1–5 estrellas |
| `ag_news` | [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news): noticias en inglés | texto, 4 temas |
| `imdb` | [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb): críticas de cine | texto, positiva/negativa |
| `electricidad` | [autogluon/chronos_datasets](https://huggingface.co/datasets/autogluon/chronos_datasets) `ercot`: demanda eléctrica por hora en Texas | serie, ventanas de 48 h |
| `tipo_cambio` | chronos_datasets `exchange_rate`: tipo de cambio diario | serie, 30 días (casi azar: muy difícil) |
| `rio` | chronos_datasets `monash_saugeenday`: caudal diario de un río | serie, 30 días |

```bash
python -m laboratorio.banco --pieza ida --tarea mnist --tarea resenas_es --guardar modelos
python -m laboratorio.banco --reales --guardar modelos      # todas (más de una hora en CPU)
```

- **Descarga y caché**: la primera vez se descargan de Hugging Face (entre 15 y 120 MB cada una). Los archivos originales quedan en `HF_HOME` y los datos ya preparados en `LABORATORIO_CACHE` (por defecto `~/.cache/laboratorio`); la segunda vez cargan en segundos. En Colab, el cuaderno 10 los guarda en Google Drive.
- **Muestra al azar**: se toman 10 000 ejemplos para entrenar y 2 000 de prueba, barajados antes de cortar (IMDB viene ordenado por clase; sin barajar saldría una sola clase).
- **Cómo se convierten**:
  - **Imágenes**: píxeles normalizados. La red densa las ve aplanadas; la CNN, con su forma.
  - **Texto**: para la red densa es un vector TF-IDF de 2048 números. Para TextCNN y Transformer son las palabras numeradas: las 20 000 más frecuentes del entrenamiento y los primeros 256 tokens.
  - **Series**: ventanas de valores pasados → predecir el siguiente. Se dividen en orden de tiempo y se normalizan solo con la parte de entrenamiento.
- `--entreno N` / `--prueba N` cambian cuántos ejemplos se usan (`--rapido` usa 3 000 / 1 000).
- **Por lotes** (128) y con **tasa de aprendizaje 0,001**, que es lo normal en conjuntos grandes.

## Parada temprana y modelos guardados

- `--paciencia N`: se separa el 15 % del entrenamiento para validación. Si en N épocas seguidas la validación no mejora, se corta y la red **vuelve a los pesos de su mejor época**. En las tareas reales viene activada con N = 5; en las pequeñas, desactivada (`--paciencia 0` la apaga). El informe JSONL guarda cuántas épocas entrenó y cuál fue la mejor.
- `--guardar carpeta`: guarda el mejor modelo de cada corrida como `pieza__tarea__s<semilla>.pt`. Se recupera con `cargar_modelo(ruta)`, que usa `torch.load(..., weights_only=True)` (solo pesos, nunca código). También guarda las activaciones mutadas de SMA, que no están en los pesos.

## Piezas (primera tanda)

`prelu_adaptativa`, `integracion_no_lineal`, `activacion_adaptativa`, `ida`, `sma`, `umbral_dinamico`, `rmb` y `hmn` vienen del cuaderno 02; `seno` viene del 04. Cada una indica de qué celda sale.

Se copiaron casi tal cual. Lo que hubo que cambiar para que funcionaran dentro del banco está marcado en el código con «Arreglo:»:
- **RMB**: la memoria cambiaba también al evaluar, y se modificaba «en el lugar» mientras el gradiente la necesitaba.
- **HMN**: tenía estados con forma variable y mensajes `print`.
- **SMA**: ahora el azar se puede repetir.
- **Umbral dinámico**: el ajuste del bias ahora lo llama el banco.

## Piezas (segunda tanda) y rivales de la industria

En `piezas_02.py`. Ganarle a ReLU ya no basta: ahora tus piezas también compiten contra lo que se usa hoy.

| Pieza | Qué es | De dónde |
|---|---|---|
| `gelu` | GELU (BERT, GPT) | industria |
| `silu` | SiLU/Swish (Llama, EfficientNet) | industria |
| `mish` | Mish (YOLOv4) | industria |
| `elu`, `leaky_relu` | ELU y LeakyReLU | industria |
| `residual` | capa con salto, estilo ResNet | industria |
| `tanh` | tanh clásica: el control justo para tus piezas que usan tanh | control |
| `memoria_errores` | recuerda las muestras con más error y las corrige tras cada paso | 02 · celdas 4-6 |
| `memoria_gradiente` | paso extra con el promedio de los gradientes recientes | 02 · celdas 8-11 |
| `lr_por_neurona` | ajuste extra con tasa por peso (un RMSprop casero) | 02 · celda 16 |
| `memoria_atencion` | mezcla gradientes recordados según el parecido de la entrada | 02 · celdas 18-19 |
| `memoria_embedding` | recuerda las pérdidas como embeddings y ajusta la capa | 02 · celdas 20-22 |
| `memoria_atencion_embedding` | igual, con atención Q/K/V sobre la memoria | 02 · celdas 23-25 |
| `inicio_perlin` | lineal + ReLU con pesos iniciales de ruido Perlin (escala He) | 02 · inicializador de pesos |
| `inicio_lorenz` | lineal + ReLU con pesos iniciales del atractor de Lorenz (escala He) | 02 · inicializador de pesos |
| `denso_cruzada` | cada capa recibe todas las anteriores (como DenseNet) | 02 · celda 32 |
| `densa_entrada` | igual, y además cada capa recibe la entrada original | 02 · celda 34 |

En los informes las de la industria salen marcadas «(industria)». `--grupo industria` o `--grupo tuyas` corre solo unas u otras.

Adaptaciones (anotadas en el código):
- Las reglas «después del paso» las llama el banco en cada lote. En el cuaderno se llamaban a mano.
- **Memoria de errores**: ahora sabe qué error tuvo cada muestra; antes había que pasárselo a mano.
- **Memoria con embeddings**: en el cuaderno era la neurona de salida de una red de regresión. Aquí es una capa oculta, para poder compararla con las demás piezas.
- **Inicializadores Perlin y Lorenz**: el ruido Perlin se calcula con NumPy (`inicializadores.ruido_perlin`) en vez de `opensimplex`. Cada semilla empieza en otro punto del ruido o de la trayectoria; si no, todas las semillas tendrían los mismos pesos y no se podría medir el azar. Como en Lorenz, la escala usa las entradas y no las salidas.
- **Denso-cruzada y densa de entrada**: no son una capa sino una forma de conectar capas, así que arman la red entera (`Pieza.red`).

- **Arreglo de inicio**: las cuatro capas con tanh (memoria de errores, de gradiente, LR por neurona y memoria con atención) empezaban con pesos `randn × 0,1`. Así quedaban atascadas en la regresión cúbica, con 60 veces más error que ReLU, y daba igual tener la regla de memoria o no. Ahora usan el mismo inicio que `nn.Linear`.

### Resultados de la segunda tanda

Informe completo: [`resultados-02-memorias.md`](resultados-02-memorias.md). Son 15 piezas en 6 tareas, con 5 semillas: 90 comparaciones corregidas por comparaciones múltiples. Resultado: 🟢 1, ⚪ 86, 🔴 3.

- **Ninguna de tus piezas de esta tanda gana** a la red estándar en ninguna tarea.
- **Memoria de errores** y **LR por neurona** salen **peores** en regresión cúbica (2–3 veces más error). LR por neurona también sale peor en XOR. Su paso extra después del optimizador estorba al optimizador (Adam), que ya hace ese trabajo.
- **Memoria de gradiente** y **memoria con atención** empatan: su paso extra se parece a un *momentum*, y Adam ya lo tiene.
- **Memoria con embeddings** (con y sin atención) y las **topologías densas** empatan, pero usan de 3 a 8 veces más parámetros.
- **Inicializadores Perlin y Lorenz** empatan en las 6 tareas: con Adam, el punto de partida casi no importa en redes de este tamaño.
- De la industria, solo **GELU** gana, y solo en la serie de seno (≈ 8 % menos error). Las demás empatan con ReLU en estas tareas pequeñas.
- **Conclusión para invertir**: en tareas pequeñas nada de esta tanda justifica más tiempo. Si quieres darles otra oportunidad, la prueba siguiente es con datos reales (`--competencia --grupo tuyas`), donde una memoria podría ayudar con datos ruidosos.

## Piezas (tercera tanda: cuaderno 02, segunda parte)

En `piezas_02b.py`.

| Pieza | Qué es | De dónde |
|---|---|---|
| `dropout` | lineal + ReLU + Dropout 10 % | industria (rival de DMNA) |
| `batchnorm`, `layernorm` | lineal + ReLU + normalización | industria (rivales de la CCN) |
| `polinomica` | términos cuadráticos y potencias de cada entrada, mezcla ReLU/tanh | 02 · neurona polinómica (4 versiones unidas) |
| `resonante` | sigmoide solo si supera un umbral de «fuerza»; si no, 0 | 02 · neurona resonante |
| `dmna` | ruido al entrenar que depende de la pérdida y se apaga con el tiempo | 02 · DMNA |
| `gmo` | mutación de todos los pesos, aceptada solo si mejora 1 % | 02 · GMO |
| `cdm` | extractor que limpia la entrada antes de la red | 02 · CDM |
| `hiper_relu` | ReLU que multiplica ×10 lo que pasa el umbral | 02 · CCN |
| `norma_caos` | normalización con «turbulencia» aprendida | 02 · CCN |
| `ccn` | la Red de Caos Controlado completa (las dos anteriores juntas) | 02 · CCN |

Arreglos (anotados en el código):
- **Polinómica**: se unieron las cuatro versiones. Se quitó el «estado», que no dependía de la entrada (era un bias más). Con muchas entradas los términos cuadráticos usan una forma de rango 16, porque la matriz completa no cabe.
- **Resonante**: en el cuaderno se entrenaba con mutaciones al azar; aquí, con gradiente. El método de mutación se comparará con los optimizadores en el cuaderno 08.
- **CDM**: el extractor se entrenaba con la entrada *sin ruido verdadera*, un dato que no existe en la vida real y que la red estándar no recibía. Ahora aprende como autoencoder de limpieza.
- **CCN**: al evaluar usaba otra escala que al entrenar, y sus estadísticas casi no se actualizaban (decaimiento 0,999). Ahora usa la misma fórmula y 0,9, como BatchNorm.
- **GMO**: compara la pérdida del **mismo lote** antes y después de mutar, con el modelo entero en modo evaluación (el banco le pasa `perdida_de`). El cuaderno usaba variables globales y comparaba con la mejor pérdida vista, que con lotes de distinta dificultad no sirve.

### Resultados de la tercera tanda

Informe completo: [`resultados-02-parte2.md`](resultados-02-parte2.md). Son 11 piezas en 6 tareas, con 5 semillas: 66 comparaciones. Resultado: 🟢 0, ⚪ 54, 🔴 12.

- **Ninguna gana** a la red estándar.
- **Polinómica**: peor en círculos, serie y dígitos. **Memoriza**: en círculos acierta 93 % al entrenar y 79 % en la prueba, y en dígitos usa 167 000 parámetros contra 3 466. En el cuaderno sí llega al 98 % en círculos con 5 neuronas, porque ese círculo tiene menos ruido y la forma x² + y² es justo un polinomio.
- **Resonante**: mucho peor en regresión (100 veces más error) y en dígitos. El umbral apaga la neurona y entonces no aprende.
- **HiperReLU**: peor en 4 tareas. Multiplicar ×10 desestabiliza el aprendizaje.
- **DMNA, GMO y CDM empatan.** GMO no aceptó ninguna mutación en 30 corridas: un ruido al azar en todos los pesos casi nunca mejora 1 % el mismo lote, así que queda igual a la red estándar.
- **Normalización con turbulencia** pierde en círculos, igual que **BatchNorm**: en redes tan pequeñas, normalizar estorba. LayerNorm empata.
- **Dropout** (industria) también sale peor en regresión cúbica: con tan pocos datos, el ruido estorba.

## Piezas (cuarta tanda: cuaderno 02, última parte)

En `piezas_02c.py`, y la CNN en `arquitecturas.py`.

| Pieza | Qué es | De dónde |
|---|---|---|
| `dendritica` | dos dendritas por neurona (ReLU y exponencial suave) y soma sigmoide | 02 · neurona dendrítica (NumPy) |
| `synaptic_tagging` | etiqueta los pesos con gradiente fuerte y los refuerza cada 10 pasos | 02 · synaptic tagging |
| `estado_interno` | tanh con un estado inicial aprendido | 02 · estado interno (TensorFlow) |
| `ruteo_adaptativo` | compuerta que reparte entre una ruta de «éxito» y una de «reevaluación» (RARC, una mezcla de expertos) | 02 · ruteo adaptativo (TensorFlow) |
| arquitectura `cnn_densa_entrada` | CNN donde cada capa recibe las anteriores y la imagen original | 02 · CNN con entrada densa |

Arreglos (anotados en el código):
- **Dendrítica**: era una sola neurona en NumPy con el gradiente calculado a mano. Ahora es una capa de PyTorch con el mismo cálculo. Usa el inicio de `nn.Linear`.
- **Estado interno** y **ruteo**: se pasaron de TensorFlow a PyTorch. El «estado» se crea de nuevo en cada llamada, así que no recuerda nada: la capa equivale a una tanh con un sesgo extra.
- **CNN con entrada densa**: en el cuaderno reconstruía MNIST (autoencoder). Aquí clasifica, igual que la CNN normal con la que compite.

### Resultados de la cuarta tanda

Informe completo: [`resultados-02-parte3.md`](resultados-02-parte3.md). Son 4 piezas en 6 tareas, con 5 semillas: 24 comparaciones. Resultado: 🟢 2, ⚪ 18, 🔴 4.

- **Synaptic tagging** y **estado interno** ganan en la serie de seno (≈ 6 % menos error). Pero la **tanh sola** da lo mismo (0,0243): la mejora viene de la tanh, no de la idea. Synaptic tagging también sale peor en XOR.
- **Dendrítica**: peor en regresión, serie y dígitos.
- **Ruteo adaptativo** empata en todo.
- **CNN con entrada densa** contra la CNN normal (dígitos 8×8, 5 semillas): 97,7 % contra 98,4 %, empate (p = 0,11), con 50 % más parámetros. Falta probarla en MNIST y CIFAR-10 con GPU (`--competencia --tipo imagen`).

## Cuaderno 03: neuronas alternativas

```bash
python -m laboratorio.banco --todo --pieza knn_prototipos --pieza genetica --pieza genetica_arreglada --pieza genetica_invertida
python -m laboratorio.banco --clasicos      # clasificadores clásicos contra la red (clasicos.md)
```

En `piezas_03.py` y `clasicos.py`.

| Pieza | Qué es |
|---|---|
| `knn_prototipos` | neuronas KNN: cada una guarda 5 patrones que aprende sin etiquetas (estilo Kohonen) y se activa según la distancia a los 3 más cercanos |
| `genetica` | redes genéticas tal cual: las neuronas con ≥ 10 ajustes se reemplazan por mutaciones de las 2 menos ajustadas |
| `genetica_arreglada` | mide cuánto se ajusta cada neurona, solo copia neuronas vivas y cambia ≤ 10 % por vez |
| `genetica_invertida` | el criterio al revés: recicla las neuronas que no se usan copiando las más útiles (como ReDo, 2023) |
| `--clasicos` | regresión logística, KNN, SVM, árbol, Naive Bayes, **Random Forest** y **Gradient Boosting** (lo que se usa hoy en datos en tabla) |

Arreglos (anotados en el código):
- **KNN**: 1/(1 + distancia) no tiene escala. Con 64 entradas todas las neuronas daban ≈ 0,1 para cualquier entrada y los patrones nunca se movían; en dígitos acertaba 7 %, menos que el azar. Ahora la distancia se mide en unidades de la distancia típica de cada neurona, como el ancho de una red RBF.
- **Neuronas con árbol de decisión y con Naive Bayes**: no se meten en una red porque necesitan una «etiqueta local» por neurona oculta que el cuaderno no define. Se miden como clasificadores completos con `--clasicos`.

### Resultados

Informes completos: [`resultados-03.md`](resultados-03.md) y [`resultados-03-clasicos.md`](resultados-03-clasicos.md).

- **Redes genéticas tal cual: se derrumban** (50 % en círculos y lunas, que es el azar). Es un fallo de diseño: en una red ReLU toda neurona viva recibe ajuste en cada paso, así que las «menos ajustadas» son las **muertas**. El algoritmo reemplaza las neuronas vivas por copias de muertas.
- **Arreglada**: va destruyendo la red poco a poco: pierde en 5 de 6 tareas (círculos 72 % contra 85 %) y en dígitos empata con mucha variación (93 % ± 4 contra 97 %). El problema de fondo es el criterio: con descenso por gradiente, la neurona que más se ajusta es la que **más importa** para el error, no una «enferma».
- **Invertida**: empata con la red estándar en las 6 tareas. Reciclar las neuronas que no se usan no daña; en redes más grandes, con muchas neuronas muertas, podría ayudar (por probar con datos reales).
- **Neuronas KNN**: pierden en todo. Solo aprende la capa de salida (33 a 330 parámetros contra más de 1 000).
- **Clasificadores clásicos**: Gradient Boosting gana en XOR y SVM gana en círculos y lunas. En la serie de seno, **una regresión lineal (Ridge) le gana a la red**: esa tarea es casi lineal. Por eso las «victorias» de IDA, seno y tanh en la serie de seno valen poco: igualan a un modelo lineal.

Limpieza del cuaderno 03 (de 86 a 28 celdas):
- Se borró la «neurona DC»: 15 versiones de una red sigmoide normal escrita con *dataclasses*, con técnicas estándar (bajar el aprendizaje por tramos o si se estanca).
- Se borraron los ejemplos de cómo cargar modelos de lenguaje (DeepSeek Coder, CodeGemma en GGUF), que no son investigación.
- Se borró el corrector GNN repetido del cuaderno 01 (ya está en el banco).
- Se borraron versiones viejas de las neuronas flotantes, del árbol fonológico y de las redes genéticas, y celdas sueltas.

## Cuaderno 04: visión (decorrelación y «papelito preguntón»)

En `piezas_04.py` y `arquitecturas.py`. El cuaderno estaba en TensorFlow; se pasó a PyTorch con el mismo cálculo.

| Pieza o arquitectura | Qué es | De dónde |
|---|---|---|
| `decorrelacion` | castiga que las neuronas de una capa se parezcan entre sí (como DeCov, 2016) | 04 · AODC-NN |
| arquitectura `cnn_decorrelada` | la CNN normal más ese castigo entre filtros de cada convolución (λ = 1e-4) | 04 · AODC-NN |
| `papelito_pregunton` | H1 extrae, H2 ve a H1, H1 se reajusta viendo a H2; se castiga que se parezcan (coseno²) | 04 · ODFE-NN y RDCR-NN |
| `papelito_covarianza` | igual, castigando la covarianza cruzada | 04 · NIDI-NN |
| `fusion_reciproca` | la misma interacción, sin castigo | 04 · RCF-NN |
| arquitectura `rcf_recurrente` | celda recurrente con H1 y H2 acoplados en cada paso, contra la LSTM | 04 · RCF-NN recurrente |

### Resultados

Informe completo: [`resultados-04.md`](resultados-04.md). Son 24 comparaciones: 🟢 0, ⚪ 22, 🔴 2.

- **Decorrelación** en capa densa: empata en las 6 tareas.
- **Papelito preguntón** (las tres versiones): empata en casi todo y pierde en círculos (79-80 % contra 85 %). Usa 5 veces más parámetros y **memoriza**: acierta 88-89 % al entrenar en círculos.
- **CNN auto-decorrelada** contra la CNN normal (dígitos, 5 semillas): **empate**, 97,9 % contra 98,4 % (p = 0,14). Una primera versión medía la covarianza antes de BatchNorm y salía peor (97,1 %): la red podía bajar la pérdida achicando pesos sin decorrelacionar de verdad (lo encontró Codex). Ahora se mide después de la activación, como en el cuaderno.
- **Celda recurrente RCF** contra la LSTM (serie de seno): **peor**, 0,0282 contra 0,0256 (p = 0,01). Contra la CNN 1D, empate.
- Falta probar las dos arquitecturas con MNIST, CIFAR-10 y series reales (con GPU): `--competencia --tipo imagen` y `--tipo serie`.

Limpieza del cuaderno 04 (de 58 a 46 celdas): se unieron las versiones de AODC (2), del papelito preguntón (RDCR ×3, NIDI y ODFE) y de RCF (3). Se quitó una celda que repetía la «densa de entrada» del cuaderno 02. La parte generativa (tripletas, interpolación de video, GAN) se mide al final, junto con los demás generativos.

## Cuaderno 05: atención

En `piezas_05.py` y `arquitecturas.py`.

| Pieza o arquitectura | Qué es | De dónde |
|---|---|---|
| arquitectura `transformer_mascara_rl` | Transformer normal donde cada cabeza aprende una máscara de posiciones T×T con ruido Gumbel, que **multiplica** los puntajes (tal cual) | 05 · atención suave con RL |
| arquitectura `transformer_mascara_rl_log` | igual, pero **suma** log(máscara) (arreglo) | 05 · atención suave con RL |
| `atencion_salida` | AttentiveLinear: cada neurona reparte su salida con factores sigmoide(A) | 05 · atención a la salida |
| `atencion_local` | agrega el promedio de entradas pasadas pesado por 1/error | 05 · atención local (TensorFlow) |

Fallos de diseño encontrados (anotados en el código):
- **Atención suave con RL**: la máscara **multiplica** los puntajes antes del softmax. Una máscara ≈ 0 no bloquea esa posición, solo vuelve el reparto más parejo (una prueba lo muestra). Sumar log(máscara) sí la bloquea. Además, pese al nombre, no usa refuerzo: el ruido Gumbel deja que la máscara aprenda con el gradiente normal.
- **AttentiveLinear**: los «factores de atención» son una matriz fija, no dependen de la entrada. La capa equivale a una lineal con pesos entre 0 y 1 y sin sesgo. Como todos empiezan en ≈ 0,5, todas las neuronas de salida dan casi lo mismo y la red no logra separarlas.
- **Atención local**: el cuaderno le pasaba siempre la pérdida de relleno 1,0, así que el peso por error nunca actuaba. Aquí recibe la pérdida real. Al evaluar, la memoria queda fija y solo agrega un vector constante.

### Resultados

Informe completo: [`resultados-05.md`](resultados-05.md).

- **Transformer con máscara** contra el Transformer normal (reseñas en español, 5 clases, 3 semillas): **empate** en las dos versiones (original 37,9 %, arreglada 38,6 %, normal 37,0 %), con casi el doble de parámetros (1,1 millones contra 0,6). Falta probarlo con más datos y GPU (`--competencia --tipo texto`).
- **AttentiveLinear**: se derrumba en todas las tareas (50 % en círculos y lunas, que es el azar; 26 % en dígitos).
- **Atención local**: empata en 5 tareas y pierde en XOR.

Limpieza del cuaderno 05 (de 37 a 13 celdas):
- La atención multi-cabeza normal tenía 4 versiones; se dejó una, como referencia.
- La atención suave con RL tenía 4 versiones para texto y 11 para imágenes; se dejaron la final de texto y dos de imágenes (con tus imágenes y con MNIST).
- Se quitaron un AttentiveLinear repetido, copias del synaptic tagging y de la neurona dendrítica (ya en el banco desde el 02) y una instalación suelta.

## Cuaderno 06: refuerzo y cerebro modular

Piezas en `piezas_06.py`: `rl_interno` (RlNeuron), `votacion_cabezas` (DecisionVotingNet), `biomas` (NeuralBiomesModel), `cerebro_moe` y `cerebro_moe_arreglado` (CorticalMoE con REINFORCE), `talamo_top2` (ThalamusBiomeNet) y, como industria, `moe_suave` (mezcla suave de expertos). Los entornos de refuerzo del cuaderno (código Python, cubos con varios cuerpos, ReroutingNetwork) quedan para la parte final.

### Resultados

Detalle en `resultados-06.md`. Nada le gana a la tecnología actual: 48 de 49 comparaciones dan «igual». En la tarea caótica el cerebro modular sube a 73,8 %, pero con 12 veces más parámetros; una red estándar del mismo tamaño llega a 72,3 % y Dropout en la red chica a 77,4 %. El cerebro modular tal cual se rompe en regresión porque todos los ejemplos reciben la misma recompensa; con recompensa por ejemplo empata.

## Cuaderno 07: grafos (parte no generativa)

En `piezas_07.py`.

| Pieza | Qué es | De dónde |
|---|---|---|
| `anti_gradiente` | extractor → capa de inversión de gradiente (GRL, λ = 1) → clasificador | 07 · GradientReversal |

Fallo de diseño: en el cuaderno la GRL va en el **camino principal**, así que el extractor aprende a empeorar la misma tarea que el clasificador intenta resolver. En la industria (DANN, Ganin y Lempitsky, 2015) la GRL va en una **rama aparte**, delante de un clasificador de «dominio» (de qué conjunto viene cada dato). Así los rasgos sirven igual en los dos dominios sin dañar la tarea principal.

### Resultados

Informe completo: [`resultados-07.md`](resultados-07.md).

- **Anti-gradiente**: peor en 5 de 6 tareas (dígitos 37 % contra 97 %, XOR 53 % contra 98 %). En lunas cuenta como empate solo por la enorme variación entre semillas (67 % ± 32).
- Lo demás del cuaderno 07 son simulaciones y dibujos sin una tarea que medir (neuronas espaciales, visualizaciones de grafos), los modelos de grafos de la industria (GCN y GAT con Cora) y la parte generativa (HCMG, D-ReG), que se mide al final.

Limpieza del cuaderno 07 (de 51 a 23 celdas):
- Se quitaron las copias con nombres en latín de HCMG y de la inversión de gradiente.
- Se dejó un solo dibujo de grafos ramificados (había 5).
- Se quitaron las demos de BERT repetidas del cuaderno 03, la neurona resonante repetida del 02, una neurona escalón básica, una celda toda comentada y la «red inversa» (una regresión lineal al revés, sin una idea nueva).
- Se arregló una celda que empezaba con una palabra suelta («Un») y no corría.

## Cuaderno 08: votación ponderada y tarea caótica 20D

- **Tarea `caotica_20d`** (en `tareas.py`): 4 clases definidas por reglas enredadas sobre 20 números, con ruido agregado después de etiquetar, igual que en el cuaderno. Se usan 4 000 ejemplos en vez de 20 000 para que corra en CPU. Es la primera tarea del banco donde la red estándar **memoriza** mucho: acierta 99,9 % al entrenar y 70 % en la prueba.
- **Votación ponderada** (en `--clasicos`): 5 clasificadores votan con sus probabilidades, cada uno pesado por su exactitud en validación cruzada. El SVM usa `CalibratedClassifierCV`, porque `SVC(probability=True)`, que usaba el cuaderno, queda obsoleto en scikit-learn 1.11 (también se actualizó la celda del cuaderno).

### Resultados

Informe completo: [`resultados-08.md`](resultados-08.md).

**Todas las piezas en la tarea caótica** (5 semillas). Aquí sí ganan las piezas que frenan la memorización, y varias son tuyas:

| Pieza | Prueba | Veredicto | De dónde |
|---|---|---|---|
| `dropout` | 77,4 % | 🟢 | industria |
| `dendritica` | 76,1 % | 🟢 | tuya (02) |
| `integracion_no_lineal` | 75,7 % | 🟢 (con 9 veces más parámetros) | tuya (02) |
| `dmna` | 72,2 % | 🟢 | tuya (02): el ruido que depende de la pérdida |
| `genetica_invertida` | 72,2 % | 🟢 | tuya, arreglada (03) |
| red estándar | 70,0 % | — | — |
| `tanh`, `seno`, capas con memoria, `polinomica`… | 64-67 % | 🔴 | — |

- Las piezas con tanh (`tanh`, `memoria_*`, `synaptic_tagging`) y `seno` empeoran: memorizan todavía más.
- **Ojo**: todas quedan muy lejos de **Gradient Boosting (94 %)**. Para datos en tabla con reglas así, la industria usa árboles por algo.
- **Pista para invertir**: la neurona dendrítica y DMNA regularizan casi tan bien como Dropout. Vale la pena compararlas directamente con Dropout en datos reales en tabla.

**Votación ponderada**: gana a la red en dígitos y en la tarea caótica (76 % contra 70 %), pero pierde en 3 de las tareas pequeñas. Siempre queda por debajo de Gradient Boosting.

Limpieza del cuaderno 08 (de 15 a 7 celdas):
- Se quitaron los generadores de series sintéticas, que ya están en `laboratorio/series.py`.
- Se quitaron el planificador de tasa de aprendizaje (es `ReduceLROnPlateau`) y la «optimización adaptativa basada en error», que dependía de la neurona DC ya borrada del cuaderno 03.
- El detector de generalizaciones («sistema inmune cognitivo») pasó al cuaderno 09, de notas e ideas: es una regla de texto, no una red.

**Agregar una idea nueva**: escribe la capa como `nn.Module(entradas, salidas)` y añádela a `PIEZAS` en `piezas.py` (o en `piezas_02.py`). Si tiene una regla que se aplica después de cada paso (como las mutaciones de SMA o las memorias), ponle un método `despues_del_paso(perdida, error, epoca)`. Si la idea es la forma de conectar las capas, usa `red=lambda entradas, salidas, ocultas: TuRed(...)`.

## El tamaño cambia el resultado

Barrido con la red estándar y con Dropout de 8 a 128 neuronas por capa (5 semillas). Detalle y gráfico en el cuaderno `laboratorio-neuronal/12_como_funciona_cada_pieza.ipynb`.

- **Tarea caótica 20D**: la red de 8 neuronas (276 parámetros) acierta 75,6 %; la de 32 (la del banco), 70,0 %, porque memoriza (98 % al entrenar). Las piezas que «ganan» ahí (Dropout 77,4 %, dendrítica 76,1 %, DMNA, genética invertida) frenan esa memorización, pero una red estándar más chica logra casi lo mismo sin ninguna idea nueva.
- **Círculos**: 8 neuronas bastan (85,7 %); con 128 baja a 82,0 %.
- **Dígitos**: sí hacen falta neuronas (93,7 % con 8, ≈ 97 % con 16 a 64).
- **Regla**: una pieza vale la pena solo si le gana a la red estándar **de su mejor tamaño**, no solo a la de 32 neuronas.

## Primeros resultados

Informe completo: [`resultados-2026-10-28.md`](resultados-2026-10-28.md). De 54 comparaciones, ya corregidas por comparaciones múltiples: 🟢 2 mejor, ⚪ 42 igual, 🔴 10 peor.

- **Series de tiempo**: solo **IDA** y **seno** salen mejores que ReLU (≈ 7 % menos error). Es la pista que vale la pena confirmar con series reales.
- **Clasificación** (círculos, lunas, XOR, dígitos): ninguna pieza gana. La mejora que se vio en los cuadernos no se sostiene con varias semillas y datos de prueba.
- **HMN** sale **peor en todas las tareas**. En entrenamiento acierta mucho más que en prueba: su memoria se ata a la posición de cada fila, así que memoriza en vez de aprender.
- **RMB** es **inestable**: en regresión cúbica algunas semillas se disparan.
- **Integración no lineal** usa ~9 veces más parámetros y aun así no gana en ninguna tarea.

Una primera versión del banco daba 6 victorias en series. Dos errores las inflaban, y una revisión automática (Codex) ayudó a encontrarlos:
1. Las ventanas de la serie se repartían al azar, y así el futuro se filtraba al entrenamiento.
2. Faltaba corregir por hacer 54 comparaciones a la vez.

Ahora la serie se divide en orden de tiempo y los veredictos usan la corrección de Benjamini-Hochberg.

## Cómo leer el informe

- **Pieza (prueba)** frente a **Estándar (prueba)**: media ± desviación entre semillas.
- **Pieza en entreno**: la misma métrica en los datos de entrenamiento. Si es mucho mejor que en prueba, la pieza **memoriza** en vez de aprender.
- **Parámetros**: una pieza con muchos más parámetros que la estándar tiene ventaja. Una victoria así hay que confirmarla contra una red estándar más grande.
- En MSE la comparación se hace con logaritmos. Así una sola semilla desastrosa no esconde que la pieza es peor en todas.
- «Igual» no es un fracaso: significa que la idea no aporta en esa tarea, y eso también es un resultado.

## Segunda tanda (pendiente)

Las piezas que necesitan su propio ciclo de entrenamiento quedan para después, porque cada una exige adaptar el banco:
- memoria de errores con hooks;
- mutación global (GMO);
- algoritmos genéticos del cuaderno 03;
- neurona DC.

Los modelos de difusión (que generan imágenes) quedan fuera por ahora: no clasifican, así que necesitarían otra métrica y mucha más GPU.
