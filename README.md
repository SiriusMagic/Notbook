# Notbook

Cuadernos de Google Colab de varios años, ordenados y documentados.

## Recomendaciones: ¿dónde usar tus piezas?

Todo está en **[`RECOMENDACIONES.md`](./RECOMENDACIONES.md)**. En corto:

- **Datos en tabla con ruido**: es donde mejor les va a tus piezas. La **neurona dendrítica** es la pista más fuerte (76 % contra 70 % de la red estándar; casi como Dropout). Le siguen DMNA y la genética invertida. Para un proyecto real en tablas, Gradient Boosting sigue siendo mejor (94 %).
- **Corrección ortográfica**: el corrector GNN **con conexiones residuales** mejora el texto con muy pocos parámetros.
- **Series con ciclos**: IDA y la activación seno bajan ≈ 7 % el error frente a ReLU, pero una regresión lineal les gana en esa serie. Falta probarlas con series reales.
- **Al final del archivo**: las piezas que no están listas o dejaron mucho que desear, con el motivo de cada una.

## Contenido

| Carpeta | Qué es |
|---|---|
| [`laboratorio-neuronal/`](./laboratorio-neuronal) | Experimentos con redes neuronales en 9 cuadernos por tema: texto, capas y activaciones, neuronas alternativas, visión, atención, refuerzo, grafos, optimización y notas. Más los cuadernos 10 y 11 (competencias con GPU) y el 12, una guía de cómo funciona cada pieza y qué resultado dio. |
| [`laboratorio/`](./laboratorio) | Piezas útiles sacadas de los cuadernos como código `.py` con pruebas, y un **banco de evaluación** (`laboratorio/banco/`) que compara tus neuronas y capas de investigación contra una red estándar: sílabas del español, series de tiempo sintéticas, programador de tasa de aprendizaje, inicializador de Lorenz, voto ponderado y capa de inversión de gradiente. |
| [`vulkan/`](./vulkan) | Portal de acceso de la red Wi-Fi Vulkan (`index.html`). |

## Qué pasó con cada cuaderno original

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

## Seguridad y mantenimiento

- **Revisión automática** (`.github/workflows/check.yml`): en cada cambio comprueba que los cuadernos tengan formato válido, que no tengan llaves de API ni resultados guardados, y avisa de errores de sintaxis. También corre las pruebas de `laboratorio/` (`pytest`).
- **Dependabot** vigila `laboratorio-neuronal/requirements.txt` y `laboratorio/requirements.txt`: avisa de fallos de seguridad y propone actualizaciones una vez al mes, agrupadas.
- `torch.load` usa `weights_only=True` (solo carga pesos, no código) y los archivos `joblib` llevan un aviso de cargar solo archivos propios.
- PyTorch Geometric se instala desde PyPI, sin instalaciones desde URLs externas.

## Abrir en Colab

**Archivo → Abrir cuaderno → GitHub** → escribe `SiriusMagic/Notbook` y elige el cuaderno. Como el repo es privado, Colab te pedirá autorizar el acceso a GitHub la primera vez.

## Licencia

Copyright 2026 SiriusMagic. Apache 2.0: consulta [`LICENSE`](./LICENSE).
