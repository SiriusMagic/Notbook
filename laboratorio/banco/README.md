# Banco de evaluación de neuronas, activaciones y capas

El cuaderno 02 tiene decenas de experimentos «mi idea contra RedEstandar». Cada uno se probó **una vez**, con una sola semilla y midiendo sobre los mismos datos de entrenamiento. Así no se puede saber si una idea es mejor o si tuvo suerte.

Este banco las mide a todas igual:

- **Mismas tareas**: las que ya usan los cuadernos, con datos fijos y una parte de **prueba** que la red nunca ve al entrenar.
- **Misma red**: dos capas ocultas de 32 neuronas y una salida lineal. Solo cambia la pieza.
- **Varias semillas** (5 por defecto). La pieza y la estándar usan las mismas semillas, así se comparan por pares.
- **Veredicto honesto** con una prueba t pareada:
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

## Piezas (primera tanda)

`prelu_adaptativa`, `integracion_no_lineal`, `activacion_adaptativa`, `ida`, `sma`, `umbral_dinamico`, `rmb` y `hmn` vienen del cuaderno 02; `seno` viene del 04. Cada una indica de qué celda sale.

Se copiaron casi tal cual. Lo que hubo que cambiar para que funcionaran dentro del banco está marcado en el código con «Arreglo:»:
- **RMB**: la memoria cambiaba también al evaluar, y se modificaba «en el lugar» mientras el gradiente la necesitaba.
- **HMN**: tenía estados con forma variable y mensajes `print`.
- **SMA**: ahora el azar se puede repetir.
- **Umbral dinámico**: el ajuste del bias ahora lo llama el banco.

**Agregar una idea nueva**: escribe la capa como `nn.Module(entradas, salidas)` y añádela a `PIEZAS` en `piezas.py`. Si tiene una regla que se aplica después de cada paso (como las mutaciones de SMA), ponle un método `despues_del_paso(perdida, error, epoca)`.

## Primeros resultados

Informe completo: [`resultados-2026-10-28.md`](resultados-2026-10-28.md). En resumen, de 54 comparaciones: 🟢 6 mejor, ⚪ 35 igual, 🔴 13 peor.

- **Series de tiempo suaves**: IDA, activación adaptativa, umbral dinámico, RMB, seno y PReLU son **mejores** que ReLU en `serie_seno` (entre 7 y 16 % menos error). Es la pista más prometedora.
- **En clasificación** (círculos, lunas, dígitos), casi todas quedan **igual** que la red estándar: la mejora que se vio en los cuadernos no se sostiene con varias semillas y datos de prueba.
- **HMN** sale **peor en todas las tareas**. En entrenamiento acierta mucho más que en prueba: su memoria se ata a la posición de cada fila, así que memoriza en vez de aprender.
- **RMB** es **inestable**: en regresión cúbica y lunas algunas semillas se disparan (± muy alto).
- **Integración no lineal** usa ~9 veces más parámetros y aun así no gana en ninguna tarea.

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

También falta MNIST completo. Por ahora `digitos` es su versión pequeña.
