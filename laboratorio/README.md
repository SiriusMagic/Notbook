# laboratorio — piezas reutilizables

Partes de los cuadernos de [`laboratorio-neuronal/`](../laboratorio-neuronal) que valía la pena conservar como código normal: archivos `.py` con pruebas automáticas, que se pueden importar desde cualquier cuaderno o proyecto.

| Módulo | Qué hace | Cuaderno de origen | Qué se corrigió |
|---|---|---|---|
| `silabas.py` | Separa palabras en sílabas (`silabear("instante") → "ins-tan-te"`), etiqueta cortes para entrenar y convierte texto en sílabas e ids | 01 · Segmentador y tokenizador de sílabas | El original fallaba con diptongos, hiatos, «qu/gu», «ü» y grupos como «ns-t» |
| `series.py` | Series de tiempo sintéticas: tendencia, estacional, AR(p) y combinada | 08 · Series temporales sintéticas | Semilla para repetir resultados; solo NumPy |
| `entrenamiento.py` | `ProgramadorLR`: baja la tasa de aprendizaje cuando el modelo se estanca | 08 · Programador dinámico de tasa de aprendizaje | Pasa a ser una clase que guarda su estado |
| `inicializadores.py` | Pesos iniciales sacados del atractor de Lorenz (sistema caótico) o de ruido Perlin, con escala He | 02 · Inicializador con Perlin y Lorenz | Calculaba la escala con las salidas en vez de las entradas |
| `ensamble.py` | Voto ponderado de varios clasificadores | 08 · `weighted_vote` | Ya no carga modelos con `joblib.load` (puede ejecutar código ajeno) |
| `banco/` | **Banco de evaluación**: mide cada idea de los cuadernos 01-08 contra la red estándar y contra la industria, con varias semillas, datos de prueba aparte y veredicto honesto. Cómo funciona: [`banco/README.md`](banco/README.md); resultados: [`banco/informes/`](banco/informes) | 01 a 08 · todos los experimentos «X contra RedEstandar» | Cada experimento se había probado una sola vez, sin datos de prueba aparte |
| `capas_torch.py` | Capa de inversión de gradiente (GRL) para PyTorch | 07 · Inversión de gradiente | Lambda como número y salida como copia |

## Uso

```python
>>> from laboratorio.silabas import silabear
>>> silabear("pingüino")
'pin-güi-no'
>>> from laboratorio.series import serie_ar
>>> serie_ar(5, [0.8], semilla=1).shape
(5,)

```

En Colab, después de clonar el repo: `import sys; sys.path.append("/content/Notbook")`.

## Pruebas

```bash
pip install -r laboratorio/requirements.txt
pytest
```

Se ejecutan en GitHub en cada cambio (`.github/workflows/check.yml`, trabajo `pruebas`).

## Qué se quedó en los cuadernos

Los cuadernos guardan el código original de cada idea. Las versiones medidas (tal cual y arregladas) están en `banco/`, y qué conviene usar en cada campo, en [`RECOMENDACIONES.md`](../RECOMENDACIONES.md).
