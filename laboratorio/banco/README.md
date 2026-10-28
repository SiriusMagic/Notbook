# Banco de evaluación de neuronas, activaciones y capas

El cuaderno 02 tiene decenas de experimentos «mi idea contra RedEstandar». Cada uno se probó **una vez**, con una sola semilla y midiendo sobre los mismos datos de entrenamiento. Así no se puede saber si una idea es mejor o si tuvo suerte.

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
| `denso_cruzada` | cada capa recibe todas las anteriores (como DenseNet) | 02 · celda 32 |
| `densa_entrada` | igual, y además cada capa recibe la entrada original | 02 · celda 34 |

En los informes las de la industria salen marcadas «(industria)». `--grupo industria` o `--grupo tuyas` corre solo unas u otras.

Adaptaciones (anotadas en el código):
- Las reglas «después del paso» las llama el banco en cada lote. En el cuaderno se llamaban a mano.
- **Memoria de errores**: ahora sabe qué error tuvo cada muestra; antes había que pasárselo a mano.
- **Memoria con embeddings**: en el cuaderno era la neurona de salida de una red de regresión. Aquí es una capa oculta, para poder compararla con las demás piezas.
- **Denso-cruzada y densa de entrada**: no son una capa sino una forma de conectar capas, así que arman la red entera (`Pieza.red`).

- **Arreglo de inicio**: las cuatro capas con tanh (memoria de errores, de gradiente, LR por neurona y memoria con atención) empezaban con pesos `randn × 0,1`. Así quedaban atascadas en la regresión cúbica, con 60 veces más error que ReLU, y daba igual tener la regla de memoria o no. Ahora usan el mismo inicio que `nn.Linear`.

### Resultados de la segunda tanda

Informe completo: [`resultados-02-memorias.md`](resultados-02-memorias.md). Son 15 piezas en 6 tareas, con 5 semillas: 90 comparaciones corregidas por comparaciones múltiples. Resultado: 🟢 1, ⚪ 86, 🔴 3.

- **Ninguna de tus piezas de esta tanda gana** a la red estándar en ninguna tarea.
- **Memoria de errores** y **LR por neurona** salen **peores** en regresión cúbica (2–3 veces más error). LR por neurona también sale peor en XOR. Su paso extra después del optimizador estorba al optimizador (Adam), que ya hace ese trabajo.
- **Memoria de gradiente** y **memoria con atención** empatan: su paso extra se parece a un *momentum*, y Adam ya lo tiene.
- **Memoria con embeddings** (con y sin atención) y las **topologías densas** empatan, pero usan de 3 a 8 veces más parámetros.
- De la industria, solo **GELU** gana, y solo en la serie de seno (≈ 8 % menos error). Las demás empatan con ReLU en estas tareas pequeñas.
- **Conclusión para invertir**: en tareas pequeñas nada de esta tanda justifica más tiempo. Si quieres darles otra oportunidad, la prueba siguiente es con datos reales (`--competencia --grupo tuyas`), donde una memoria podría ayudar con datos ruidosos.

**Agregar una idea nueva**: escribe la capa como `nn.Module(entradas, salidas)` y añádela a `PIEZAS` en `piezas.py` (o en `piezas_02.py`). Si tiene una regla que se aplica después de cada paso (como las mutaciones de SMA o las memorias), ponle un método `despues_del_paso(perdida, error, epoca)`. Si la idea es la forma de conectar las capas, usa `red=lambda entradas, salidas, ocultas: TuRed(...)`.

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
