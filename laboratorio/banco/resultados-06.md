# Cuaderno 06: refuerzo y cerebro modular

Piezas en `laboratorio/banco/piezas_06.py`. Para repetir: `python -m laboratorio.banco --pieza rl_interno --pieza votacion_cabezas --pieza biomas --pieza cerebro_moe --pieza cerebro_moe_arreglado --pieza talamo_top2 --pieza moe_suave --semillas 5`.

## Conclusión

- **Nada del 06 le gana a la tecnología actual.** En 48 de 49 comparaciones el resultado es «igual» a la red estándar (después de corregir por comparaciones múltiples). La única «victoria» (votación de cabezas en la serie de seno) vale poco: en esa tarea una regresión lineal le gana a todas las redes (ver resultados-02).
- **Tarea caótica 20D (la red estándar memoriza):** varias piezas suben un poco, pero tienen de 12 a 20 veces más parámetros. Comparación justa con una red estándar del mismo tamaño (5 semillas):

| Modelo | Parámetros | Prueba |
|---|---|---|
| Dropout, red chica (industria) | 1 860 | **77,4 %** |
| Dropout, red grande (industria) | 23 244 | 75,2 % |
| cerebro modular (CorticalMoE, tal cual) | 23 428 | 73,8 % |
| red estándar grande | 23 244 | 72,3 % |
| biomas | 38 980 | 71,8 % |
| cerebro modular arreglado | 23 428 | 71,7 % |
| tálamo top-2 (con entropía) | 29 924 | 70,4 % |
| red estándar chica | 1 860 | 70,0 % |
| mezcla suave de expertos (industria) | 5 556 | 68,8 % |

  El cerebro modular tal cual gana 1,5 puntos a la estándar de su tamaño, probablemente porque elegir un experto al azar al entrenar actúa como ruido (como Dropout): es la única pieza que no llega al 100 % en entrenamiento (91 %). Dropout hace lo mismo mejor y sin 20 000 parámetros extra.
- **El cerebro modular tal cual se rompe en regresión** (MSE 0,097 contra 0,005): la recompensa es la pérdida del lote entero, la misma para todos los ejemplos, así que el router no sabe qué elección fue buena. Con recompensa por ejemplo y línea base (la versión arreglada) empata.
- **«RlNeuron» no es refuerzo**: su factor solo crece con el tamaño del gradiente. Empata en todo.
- **La entropía del tálamo** (tu nota: «maximizar entropía para forzar especialización») no ayuda: 70,4 % con ella contra 71,4 % sin ella en la tarea caótica.
- **La LSTM de los biomas y la atención del experto «contexto» no hacen nada como memoria**: reciben secuencias de largo 1. Para que sirvan hace falta darles secuencias reales (por ejemplo, la serie completa, no solo la ventana aplanada).

## Arreglos anotados

- RlNeuron: pesos con el inicio de `nn.Linear` (el cuaderno usaba `randn × 0,1` y con el factor 0,1 las salidas quedaban casi en cero).
- CorticalMoE: al evaluar se usa el experto más probable (el cuaderno sorteaba también al evaluar, así que la misma entrada daba respuestas distintas).
- ThalamusBiomeNet: la «retroalimentación» (salida anterior) ya no cambia al evaluar; antes la predicción de un ejemplo dependía de los ejemplos evaluados antes. La entropía del router ahora entra en la pérdida.
- La red de comparación del cuaderno (StandardNet) tenía un error (`h2 = F.relu(self.fc2(h2))`); el banco usa su propia red estándar, así que no afecta estas cifras.

## Tabla completa

Cada celda: media ± desviación en **datos de prueba**, con 5 semillas. Exactitud: más es mejor. MSE: menos es mejor. Veredicto con prueba t pareada corregida por comparaciones múltiples (Benjamini-Hochberg, 5 %). Es una exploración: un 🟢 indica una pista que vale la pena confirmar, no una prueba definitiva.

| Pieza | Tarea | Pieza (prueba) | Estándar (prueba) | Veredicto | Pieza en entreno | Parámetros (pieza / estándar) |
|---|---|---|---|---|---|---|
| rl_interno | regresion_cubica | 0.0050 ± 0.0003 | 0.0048 ± 0.0002 | ⚪ igual | 0.0033 | 1,217 / 1,153 |
| votacion_cabezas | regresion_cubica | 0.0048 ± 0.0001 | 0.0048 ± 0.0002 | ⚪ igual | 0.0032 | 4,420 / 1,153 |
| biomas | regresion_cubica | 0.0055 ± 0.0015 | 0.0048 ± 0.0002 | ⚪ igual | 0.0040 | 37,569 / 1,153 |
| cerebro_moe | regresion_cubica | 0.0971 ± 0.0733 | 0.0048 ± 0.0002 | ⚪ igual | 0.0792 | 13,644 / 1,153 |
| cerebro_moe_arreglado | regresion_cubica | 0.0061 ± 0.0014 | 0.0048 ± 0.0002 | ⚪ igual | 0.0039 | 13,644 / 1,153 |
| talamo_top2 | regresion_cubica | 0.0051 ± 0.0005 | 0.0048 ± 0.0002 | ⚪ igual | 0.0034 | 28,716 / 1,153 |
| moe_suave (industria) | regresion_cubica | 0.0052 ± 0.0012 | 0.0048 ± 0.0002 | ⚪ igual | 0.0037 | 4,552 / 1,153 |
| rl_interno | circulos | 84.0 ± 0.5 % | 84.8 ± 0.6 % | ⚪ igual | 84.3 | 1,282 / 1,218 |
| votacion_cabezas | circulos | 84.4 ± 1.0 % | 84.8 ± 0.6 % | ⚪ igual | 84.2 | 4,584 / 1,218 |
| biomas | circulos | 80.0 ± 2.1 % | 84.8 ± 0.6 % | ⚪ igual | 93.8 | 37,698 / 1,218 |
| cerebro_moe | circulos | 82.1 ± 4.1 % | 84.8 ± 0.6 % | ⚪ igual | 81.3 | 14,324 / 1,218 |
| cerebro_moe_arreglado | circulos | 84.2 ± 2.1 % | 84.8 ± 0.6 % | ⚪ igual | 84.3 | 14,324 / 1,218 |
| talamo_top2 | circulos | 82.0 ± 1.4 % | 84.8 ± 0.6 % | ⚪ igual | 87.3 | 28,948 / 1,218 |
| moe_suave (industria) | circulos | 84.0 ± 1.4 % | 84.8 ± 0.6 % | ⚪ igual | 85.6 | 4,716 / 1,218 |
| rl_interno | lunas | 96.4 ± 0.3 % | 96.3 ± 0.3 % | ⚪ igual | 98.0 | 1,282 / 1,218 |
| votacion_cabezas | lunas | 96.1 ± 0.0 % | 96.3 ± 0.3 % | ⚪ igual | 97.7 | 4,584 / 1,218 |
| biomas | lunas | 96.4 ± 0.5 % | 96.3 ± 0.3 % | ⚪ igual | 98.5 | 37,698 / 1,218 |
| cerebro_moe | lunas | 95.6 ± 1.3 % | 96.3 ± 0.3 % | ⚪ igual | 96.5 | 14,324 / 1,218 |
| cerebro_moe_arreglado | lunas | 96.7 ± 0.0 % | 96.3 ± 0.3 % | ⚪ igual | 98.1 | 14,324 / 1,218 |
| talamo_top2 | lunas | 96.0 ± 0.5 % | 96.3 ± 0.3 % | ⚪ igual | 98.0 | 28,948 / 1,218 |
| moe_suave (industria) | lunas | 96.3 ± 0.3 % | 96.3 ± 0.3 % | ⚪ igual | 98.0 | 4,716 / 1,218 |
| rl_interno | xor | 98.0 ± 0.5 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 1,282 / 1,218 |
| votacion_cabezas | xor | 97.3 ± 0.4 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 4,584 / 1,218 |
| biomas | xor | 97.5 ± 0.0 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 37,698 / 1,218 |
| cerebro_moe | xor | 93.8 ± 3.0 % | 98.2 ± 0.4 % | ⚪ igual | 96.6 | 14,324 / 1,218 |
| cerebro_moe_arreglado | xor | 97.5 ± 0.0 % | 98.2 ± 0.4 % | ⚪ igual | 99.9 | 14,324 / 1,218 |
| talamo_top2 | xor | 97.8 ± 0.5 % | 98.2 ± 0.4 % | ⚪ igual | 99.9 | 28,948 / 1,218 |
| moe_suave (industria) | xor | 97.7 ± 0.4 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 4,716 / 1,218 |
| rl_interno | serie_seno | 0.0261 ± 0.0005 | 0.0261 ± 0.0007 | ⚪ igual | 0.0219 | 1,569 / 1,505 |
| votacion_cabezas | serie_seno | 0.0244 ± 0.0005 | 0.0261 ± 0.0007 | 🟢 mejor | 0.0192 | 4,772 / 1,505 |
| biomas | serie_seno | 0.0276 ± 0.0009 | 0.0261 ± 0.0007 | ⚪ igual | 0.0149 | 38,273 / 1,505 |
| cerebro_moe | serie_seno | 0.0262 ± 0.0015 | 0.0261 ± 0.0007 | ⚪ igual | 0.0199 | 18,968 / 1,505 |
| cerebro_moe_arreglado | serie_seno | 0.0273 ± 0.0008 | 0.0261 ± 0.0007 | ⚪ igual | 0.0174 | 18,968 / 1,505 |
| talamo_top2 | serie_seno | 0.0302 ± 0.0032 | 0.0261 ± 0.0007 | ⚪ igual | 0.0142 | 29,068 / 1,505 |
| moe_suave (industria) | serie_seno | 0.0259 ± 0.0009 | 0.0261 ± 0.0007 | ⚪ igual | 0.0165 | 4,904 / 1,505 |
| rl_interno | digitos | 96.7 ± 0.5 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 3,530 / 3,466 |
| votacion_cabezas | digitos | 97.2 ± 0.5 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 7,624 / 3,466 |
| biomas | digitos | 96.2 ± 0.9 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 42,186 / 3,466 |
| cerebro_moe | digitos | 97.0 ± 0.4 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 45,900 / 3,466 |
| cerebro_moe_arreglado | digitos | 97.3 ± 0.2 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 45,900 / 3,466 |
| talamo_top2 | digitos | 96.5 ± 0.5 % | 97.1 ± 0.5 % | ⚪ igual | 99.9 | 32,532 / 3,466 |
| moe_suave (industria) | digitos | 96.4 ± 0.6 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 7,756 / 3,466 |
| rl_interno | caotica_20d | 69.0 ± 1.3 % | 70.0 ± 0.9 % | ⚪ igual | 97.2 | 1,924 / 1,860 |
| votacion_cabezas | caotica_20d | 68.8 ± 2.5 % | 70.0 ± 0.9 % | ⚪ igual | 100.0 | 5,424 / 1,860 |
| biomas | caotica_20d | 71.7 ± 0.7 % | 70.0 ± 0.9 % | ⚪ igual | 100.0 | 38,980 / 1,860 |
| cerebro_moe | caotica_20d | 73.8 ± 1.1 % | 70.0 ± 0.9 % | ⚪ igual | 90.7 | 23,428 / 1,860 |
| cerebro_moe_arreglado | caotica_20d | 71.7 ± 0.8 % | 70.0 ± 0.9 % | ⚪ igual | 100.0 | 23,428 / 1,860 |
| talamo_top2 | caotica_20d | 70.4 ± 2.1 % | 70.0 ± 0.9 % | ⚪ igual | 98.2 | 29,924 / 1,860 |
| moe_suave (industria) | caotica_20d | 68.8 ± 0.7 % | 70.0 ± 0.9 % | ⚪ igual | 100.0 | 5,556 / 1,860 |

**Resumen:** 🟢 mejor: 1, ⚪ igual: 48, 🔴 peor: 0, 💥 falla: 0

«Pieza en entreno» mucho mejor que «Pieza (prueba)» = la pieza memoriza en vez de aprender.
Las piezas marcadas «(industria)» (GELU, SiLU, Mish…) son la tecnología de hoy: si tu pieza no les gana a ellas, no conviene invertir en ella aunque le gane a ReLU.
Ojo: una pieza con muchos más parámetros que la estándar tiene ventaja; compárala también con una estándar más grande.
