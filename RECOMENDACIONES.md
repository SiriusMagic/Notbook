# Recomendaciones: dónde conviene usar tus piezas

Resumen de todo lo que se midió en el banco de evaluación (`laboratorio/banco/`). Cada pieza se probó con 5 semillas, con datos de **prueba** que la red no vio al entrenar, contra la red estándar y contra la tecnología de la industria. Los detalles están en [`laboratorio/banco/informes/`](./laboratorio/banco/informes), y el cuaderno `laboratorio-neuronal/12_como_funciona_cada_pieza.ipynb` explica cómo funciona cada una.

> **Regla de oro.** Ninguna pieza tuya le gana hoy a la tecnología de la industria con datos que la red nunca vio. Lo más cerca son las **tripletas con flujo** (video), que ya empatan con la U-Net usando 19 veces menos parámetros, y el **stacking**, que es lo mejor medido en dígitos. Lo demás son **pistas para seguir investigando** en un campo concreto, no piezas listas para producción. Antes de invertir en una, debe ganarle también a la red estándar **de su mejor tamaño** (a veces una red más chica rinde más) y a lo que usa la industria en ese campo.

## 1. Recomendadas por campo

Ordenadas de la pista más fuerte a la más débil.

### Video: inventar fotogramas intermedios

| Pieza | Resultado | Recomendación |
|---|---|---|
| **Tripletas con flujo** (04, mejoradas) | **19,22 dB con 24 mil parámetros**; la U-Net de la industria, 19,13 dB con 467 mil (19 veces más). Con 4 fotogramas: 20,25 contra 19,79 dB | **La pista más fuerte del laboratorio: empatan con la U-Net** (o la superan un poco en PSNR; la U-Net sigue algo más nítida) con 19 veces menos parámetros y la mitad del tiempo de entrenamiento. **Recomendadas donde la memoria o la batería cuentan** (celular, cámaras, video en vivo). Falta probarlas con video real. |
| Tripletas dilatadas (04, mejoradas) | 18,95 dB con 17 mil parámetros | Una opción todavía más chica si no se puede usar `grid_sample`. |
| Tripletas tal cual (04) | 18,32 dB con 11 mil parámetros | Superadas por las dos versiones mejoradas. |

### Datos en tabla con ruido (pocas filas, muchas columnas, riesgo de memorizar)

Es el campo donde mejor les fue a tus piezas. Ahí la red estándar memoriza (acierta 98 % al entrenar y 70 % en la prueba), y ganan las piezas que frenan esa memorización.

| Pieza | Acierto en prueba (tarea caótica 20D) | Recomendación |
|---|---|---|
| **Neurona dendrítica** (02) | 76,1 % (red estándar: 70,0 %) | **La pista más fuerte.** Casi iguala a Dropout (77,4 %) con los mismos parámetros. Úsala como capa que regulariza en redes para datos en tabla. Próximo paso: compararla con Dropout en tablas reales. |
| **DMNA** (02) | 72,2 % | Ruido que depende de la pérdida. Ayuda poco, pero no daña. Sirve como alternativa a Dropout cuando conviene un ruido que se apague solo con el tiempo. |
| **Base radial entrenable** (03, reparada: antes neuronas KNN) | 76,9 % (red estándar: 70,5 %) | **Tan fuerte como la dendrítica**, ver sección 4. Probarla con tablas reales. |
| **Genética invertida** (03, arreglada) | 72,2 % | Recicla las neuronas que no se usan (como ReDo, 2023). Conviene en **redes grandes con muchas neuronas muertas**: en redes chicas no hay mucho que reciclar. |

Ojos abiertos:
- Una red estándar de solo 8 neuronas llega a 75,6 % en esta misma tarea. Parte de la ventaja viene de frenar la memorización, y eso también se logra achicando la red.
- En datos en tabla la industria usa árboles: **Gradient Boosting llega a 94 %**, muy por encima de cualquier red. Para un proyecto real en tablas, empieza por Gradient Boosting.

### Combinar varios modelos (ensambles)

| Pieza | Resultado | Recomendación |
|---|---|---|
| **Stacking** (recuperado de tu cuaderno completo) | Lo mejor en dígitos (98,5 %); en la tarea caótica 83,6 % (red estándar 70,0 %) | **Recomendado** cuando ya tienes varios modelos y quieres juntarlos. Supera a la votación ponderada en 4 de 5 tareas. |
| Votación ponderada (08) | 98,3 % en dígitos y 76,3 % en la caótica, pero pierde en XOR, lunas y círculos | Usa el stacking en su lugar: es la versión que aprende los pesos. |

### Corrección ortográfica de texto

| Pieza | Resultado | Recomendación |
|---|---|---|
| **Corrector GNN con conexiones residuales** (01) | Baja las letras mal de 12,6 % a 10,8 %; el diccionario clásico (Norvig/SymSpell) llega a 6,7 % | **Recomendado para corregir errores de letras** en textos cortos, donde un modelo grande sería caro. Sin conexiones residuales olvida la letra original y empeora el texto: usa siempre la versión con residuales. Falta medirlo en Colab con GPU contra la BiLSTM y el Transformer de letras (cuaderno 11). |

### Series de tiempo periódicas (ondas, ciclos)

| Pieza | Resultado | Recomendación |
|---|---|---|
| **IDA** y **seno** como activación (02 y 04) | ≈ 7 % menos error que ReLU en la serie de seno | Solo para series con ciclos claros. **Ojo:** en esa misma serie una regresión lineal (Ridge) les gana a todas las redes, así que la victoria vale poco. Probarlas con series reales (electricidad, ríos) en el cuaderno 10 antes de usarlas. |
| **GELU** (industria) | ≈ 8 % menos error que ReLU en la misma serie | Si quieres una activación distinta de ReLU para series, GELU es la opción probada. |

### Texto en español: tokenización

| Pieza | Resultado | Recomendación |
|---|---|---|
| **Sílabas por reglas** (01) | 2 tokens por palabra, 0,4 % de desconocidas. Al clasificar reseñas con una CNN: **39,9 %** contra 35,2 % de BPE y 34,7 % de GPT-4o (sin diferencia significativa con 2 semillas) | Interesante para el español **con pocos datos**: respeta la pronunciación y rinde igual o un poco mejor que BPE. Con muchos datos y modelos grandes, BPE da secuencias más cortas (1,2 tokens por palabra). Falta medir con el Transformer en GPU (cuaderno 13). |
| **Segmentador BiLSTM de sílabas** (01) | 99,2 % de palabras exactas (pyphen: 71,1 %) | Recomendado si se quiere silabear aprendiendo de ejemplos (otro idioma, nombres raros). Para el español, las reglas escritas a mano son igual de exactas y 80 veces más rápidas. |
| **Tokenizador con refuerzo, arreglado** (01) | Corta casi igual que BPE (1,23 contra 1,19 tokens por palabra), con 6 veces más desconocidas | Funciona, pero no aporta nada sobre BPE, que es más rápido y está probado. Solo como experimento. |

### Generar texto

| Pieza | Resultado | Recomendación |
|---|---|---|
| **Difusión de texto con máscaras** (01: tu versión de la celda 127 y la arreglada como MDLM/LLaDA) | Tu versión rellena huecos igual que la de la industria (3,0 % contra 3,4 %); para generar hay que sortear las palabras en vez de elegir siempre la más probable. Genera palabras variadas, pero el Transformer autorregresivo (como GPT) del mismo tamaño gana (perplejidad 263 contra 803) | Úsala solo si necesitas **rellenar huecos en medio de un texto**, algo que un modelo palabra por palabra no hace. Para escribir texto de corrido, usa un Transformer autorregresivo. Necesita GPU y mucho más entrenamiento para competir. |

### Agentes de refuerzo (lecciones del cuaderno 06)

| Qué | Resultado | Recomendación |
|---|---|---|
| **Guiar al agente** (recompensa pequeña por acercarse al objetivo) | Empujar el cubo entre dos: de 0 % a **65 %** de éxito | Si la recompensa llega muy tarde, el agente no aprende nada: dale pistas intermedias. |
| **REINFORCE estándar** (recompensa futura menos el promedio) | ReroutingNetwork: llega al óptimo en las 3 semillas; tal cual, una se derrumba | Usa siempre la versión estándar (o PPO). |

## 2. Neutras: no ayudan, pero tampoco dañan

Empatan con la red estándar en casi todo. No vale la pena invertir más en ellas salvo que haya una idea nueva que probar.

- **Capas y activaciones** (02): PReLU adaptativa, activación adaptativa, umbral dinámico, SMA, memoria de gradiente, memoria con atención, memoria con embeddings, inicializadores Perlin y Lorenz, topologías densas (denso-cruzada, densa de entrada), GMO, CDM, ruteo adaptativo.
- **Integración no lineal** (02): 75,7 % en la tarea caótica, pero con 9 veces más parámetros.
- **Visión** (04): CNN auto-decorrelada (97,9 % contra 98,4 % de la CNN normal), decorrelación en capa densa, CNN con entrada densa (02). Falta probarlas con MNIST y CIFAR-10 en GPU.
- **Atención** (05): Transformer con máscara aprendida (empata con casi el doble de parámetros), atención local.
- **Cerebro modular** (06): versión arreglada, tálamo top-2, votación de cabezas, biomas, RlNeuron.
- **HCMG** (07): sus dos condiciones funcionan (sin ellas pierde 4 dB), pero una U-Net con las condiciones como canales extra le gana (19,80 contra 19,29 dB) con 3 veces menos parámetros. Una primera medición decía lo contrario por una fuga de datos (los dígitos de prueba se repetían con los de entrenamiento).
- **D-ReG** (07): con poco entrenamiento parecía el mejor generador de texto, pero una GRU del mismo tamaño le gana (perplejidad 146 contra 176).
- **Tripletas con corrección sobre el promedio** (04): da lo mismo que las tripletas tal cual.

## 3. No están listas o dejaron mucho que desear

Fallan por un error de diseño. Cada una tiene anotado en el código qué falla y, cuando se pudo, una versión arreglada.

| Pieza | Qué pasa |
|---|---|
| Redes genéticas, tal cual y arreglada (03) | Se derrumban: copian neuronas muertas encima de las vivas. Usa la versión invertida. |
| AttentiveLinear (05) | Se derrumba (azar en círculos y lunas): sus «factores de atención» no dependen de la entrada. **Reparada (sección 4): empata.** |
| Anti-gradiente (07) | Peor en 5 de 6 tareas: invierte el gradiente en el camino principal. En la industria (DANN) va en una rama aparte. **Reparada como DANN (sección 4).** |
| Neuronas KNN (03) | Pierden en todo: solo aprende la capa de salida. **Reparadas como base radial (sección 4): 🟢 prometedoras en tablas.** |
| HMN (02) | Peor en todas las tareas: memoriza la posición de cada fila. **Reparada (sección 4).** |
| RMB (02) | Inestable: algunas semillas se disparan. **Reparada (sección 4): empata.** |
| Neurona resonante (02) | El umbral la apaga y deja de aprender (100 veces más error en regresión). **Reparada (sección 4).** |
| HiperReLU y CCN (02) | Multiplicar ×10 desestabiliza el aprendizaje. **HiperReLU reparada (sección 4): empata.** |
| Polinómica (02) | Memoriza: 93 % al entrenar y 79 % en la prueba en círculos. **Reparada (sección 4): empata.** |
| Papelito preguntón (04) | 5 veces más parámetros, memoriza y pierde en círculos. |
| Celda recurrente RCF (04) | Peor que la LSTM en series. |
| Memoria de errores y LR por neurona (02) | Peores en regresión: su paso extra estorba al optimizador Adam. |
| Synaptic tagging y estado interno (02) | Su «victoria» en la serie de seno es de la tanh, no de la idea. |
| Cerebro modular tal cual (06) | Se rompe en regresión: todos los ejemplos reciben la misma recompensa. |
| Agente PPO que predice la siguiente palabra (01) | Acierta 9-12 % contra 16,7 % de la misma red entrenada de forma normal, y termina respondiendo casi siempre la misma palabra. Para predecir texto, el refuerzo no conviene. |
| Tokenizador con refuerzo, tal cual (01) | Su «calidad» es un número al azar; termina cortando casi letra por letra (3,7 tokens por palabra). |
| Difusión de texto, tal cual (01) | Solo repite una palabra («sirve sirve sirve…»): el ruido es casi nulo (aprende a copiar) y genera desde máscaras que nunca vio. |
| FrameInterpolator con LSTM (04) | 66 millones de parámetros, 2,6 dB peor que la U-Net y más borroso que copiar el fotograma vecino: la LSTM recibe la imagen aplanada y no sabe qué píxeles están cerca. Para video, usa convoluciones. **Reemplazado por las tripletas con flujo de 4 fotogramas: 20,25 dB, +3,1 dB.** |
| Escribir código con PPO (06) | El objetivo del cuaderno («hola mundo») es imposible (no hay espacio entre los caracteres). Con uno posible («hola») tampoco lo logra en 300 000 pasos: la recompensa llega demasiado tarde. Para generar código, la industria parte de un modelo que ya sabe escribir. |
| Empujar cubos sin guía (06) | 0 % de éxito: sin una pista intermedia los dos cuerpos nunca coinciden al empujar. Con guía, 65 %. |
| REINFORCE tal cual (06, ReroutingNetwork) | Inestable: según la semilla llega al óptimo o se derrumba (19,0 ± 14,2 de 35,8). |

**Falta medir con GPU y datos reales:** las tripletas con video real, HCMG con imágenes y texto reales, y las competencias de los cuadernos 10 y 11 (MNIST, CIFAR-10, reseñas, series reales).

## 4. Reparadas: las de la lista anterior, arregladas

Corregí el error de diseño de 8 piezas de la sección 3 sin cambiar su idea, y las medí otra vez junto a la original (5 semillas, 7 tareas). Detalle en [`informes/reparadas.md`](./laboratorio/banco/informes/reparadas.md); código en `laboratorio/banco/piezas_reparadas.py`.

| Pieza reparada | Qué cambió | Resultado | Recomendación |
|---|---|---|---|
| **Base radial entrenable** (`rbf_entrenable`, antes neuronas KNN) | Los centros y su ancho se aprenden con gradiente | 🟢 **76,9 %** en la tarea caótica (estándar 70,5 %, Dropout 77,4 %) y menos error en la serie de seno; antes perdía en las 7 tareas | **Nueva pista para datos en tabla con ruido**, a la par de la dendrítica. Próximo paso: probarla con tablas reales. |
| HMN sin memoria entre filas (`hmn_sin_memoria`) | La mini-LSTM arranca de cero en cada ejemplo | 🟢 gana en la serie de seno, 🔴 pierde en círculos; antes perdía en 6 de 7 | Neutra: una activación con compuertas, como una GELU aprendida. |
| Atención dinámica (`atencion_dinamica`, antes AttentiveLinear) | La atención depende de la entrada | ⚪ empata en las 7; antes perdía en 6 | Neutra. Es la idea de Squeeze-and-Excitation; rinde más en imágenes (CNN). |
| RMB estable (`rmb_estable`) | Inicio estándar y memoria acotada | ⚪ empata en las 7; ya no se dispara | Neutra. |
| Polinómica regularizada (`polinomica_regularizada`) | Rango bajo, entrada acotada y penalización | ⚪ empata en las 7, con 3 a 6 veces menos parámetros que la original | Neutra. |
| HiperReLU suave (`hiper_relu_suave`) | Amplificación aprendida desde ×1 | ⚪ empata en las 7 (aprende a no amplificar) | Neutra: el ×10 no aportaba. |
| Resonante suave (`resonante_suave`) | Umbral suave y aprendido | ⚪ empata en 6, 🔴 pierde en la caótica | Neutra. |
| Anti-gradiente como DANN (`RedDANN`) | La inversión de gradiente va en una rama que adivina el dominio | +3 puntos con dígitos de «otra cámara» (42,2 % contra 39,1 %), sin diferencia significativa | Úsala solo para **adaptar un modelo a datos nuevos sin etiquetas** (otra tienda, otra cámara) y con λ pequeño (0,03). |

La lección: casi todas fallaban por tener algo **fijo o duro** (umbral, amplificación, atención, centros) donde hacía falta algo **aprendido y suave**.

## 5. Ideas de investigación para seguir

Ordenadas por lo que más podría rendir con lo que ya está medido.

1. **Tripletas con flujo en video real** (UCF101, Vimeo-90k) y a mayor resolución, contra RIFE. Con dígitos ya le ganan a la U-Net con 19 veces menos parámetros (sección 1).
2. **Base radial + dendrítica + stacking para tablas**: las tres piezas que mejor rinden en datos con ruido, juntas como modelos base de un stacking, contra Gradient Boosting en tablas reales (OpenML).
3. **Router troncal + mini-expertos LoRA** (cuaderno 14): un clasificador pequeño elige qué adaptador LoRA de oficio usar. Falta medir en GPU si el router con LoRA supera a un solo LoRA para todos los oficios.
4. **Corrector híbrido**: el diccionario propone candidatos y el GNN con residuales elige según el contexto. Hoy el diccionario solo corrige mejor que la red (CER 0,067 contra 0,108 en sustituciones), y los dos juntos podrían sumar.
5. **Tokenizador de sílabas + BPE**: cortar en sílabas con la BiLSTM (99,2 % de palabras exactas) y aplicar BPE sobre las sílabas, para que los tokens respeten la pronunciación sin tener tantas desconocidas.
6. **Difusión por bloques**: tu difusión con máscaras para rellenar huecos, y un Transformer autorregresivo para escribir de corrido, en un mismo modelo (como *Block Diffusion*, 2025).
7. **DANN para reseñas de otra tienda**: entrenar con reseñas de Amazon y adaptar, sin etiquetas, a reseñas de otro sitio.
