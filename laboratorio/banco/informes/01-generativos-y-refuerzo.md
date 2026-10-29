# Cuaderno 01: parte generativa y de refuerzo

Tres ideas del cuaderno 01, cada una tal cual y arreglada, contra lo que usa la industria. Datos: reseñas reales en español (`resenas_es`, Hugging Face). Código: `siguiente_palabra.py`, `tokenizador_rl.py` y `difusion_texto.py`.

```bash
python -m laboratorio.banco --siguiente_palabra --entreno 3000 --epocas 15
python -m laboratorio.banco --difusion --entreno 5000 --pasos 800 --semillas 2   # en Colab con GPU: sin límites
python -m laboratorio.banco --tokenizadores --tarea resenas_es                    # incluye rl_cuaderno y rl_arreglado
```

## Conclusión

- **El «agente PPO» no segmenta: predice la siguiente palabra, y lo hace peor que lo normal.** Una red entrenada de forma supervisada (o contar pares de palabras) acierta 16,7 %; PPO, 9-12 %, y termina respondiendo casi siempre la misma palabra. Para predecir texto, el refuerzo no conviene: solo dice «acertaste / no», mientras que lo supervisado dice cuál era la palabra correcta.
- **El tokenizador con refuerzo, tal cual, corta casi letra por letra** (3,7 tokens por palabra), porque su «calidad» es un número al azar. Con una recompensa real (pocos tokens que se repitan) corta casi igual que BPE (1,2 tokens por palabra), pero con 6 veces más desconocidas. No aporta nada sobre BPE.
- **La difusión de texto del cuaderno no aprende a generar**: el ruido es tan pequeño que solo aprende a copiar, y al generar desde [MASK] repite una palabra («sirve sirve sirve…»). Tu otra versión (celda 127 del cuaderno completo) ya usaba máscaras y rellena huecos igual que la de la industria. La versión con máscaras (como MDLM y LLaDA) sí genera palabras variadas y rellena huecos, pero con este tamaño y entrenamiento el Transformer autorregresivo (como GPT) gana con claridad (perplejidad 263 contra 803). La difusión de texto necesita modelos y entrenamientos mucho más grandes (GPU) para competir.

## 1. Siguiente palabra (`--siguiente_palabra`)

3 000 reseñas para entrenar y 1 000 de prueba, vocabulario de 2 000 palabras, 3 semillas, 15 épocas.

| Modelo | Acierto (top-1) | Entre las 5 primeras | Perplejidad (menos es mejor) |
|---|---|---|---|
| misma red, supervisada (industria) | 16,7 % ± 0,1 | 40,2 % ± 0,1 | 85 |
| conteo de pares de palabras (clásico) | 16,7 % | 39,8 % | 143 |
| PPO tal cual (gamma = 0,99) | 11,7 % ± 0,4 | 12,1 % ± 0,1 | enorme (casi siempre la misma palabra) |
| PPO con gamma = 0 (variante) | 9,3 % ± 3,6 | 11,6 % ± 5,1 | enorme (casi siempre la misma palabra) |

- Fallos del cuaderno: el vocabulario tenía 14 palabras y las frases usaban otras (casi nunca podía acertar), y en el último paso pedía repetir la última palabra en vez del fin de frase. Aquí el vocabulario sale del texto.
- La variante con gamma = 0 (cada predicción juzgada solo por su propia recompensa) **no ayuda**: el problema es la señal, no el descuento.

## 2. Tokenizador con refuerzo (`rl_cuaderno`, `rl_arreglado`)

Entrenado con 3 000 reseñas y medido en 500 nuevas, con el mismo vocabulario de 8 000 unidades para todos.

| Tokenizador | Tokens por palabra (menos = más corto) | Desconocidas | Ejemplo: «el producto llegó rápido pero la calidad es pésima» |
|---|---|---|---|
| `rl_cuaderno` (tal cual) | 3,66 | 0,2 % | el · p r o d u c t o · l l egó · rá p i do … |
| `rl_arreglado` | 1,23 | 3,0 % | el · producto · llegó · rápido · pero … |
| BPE (industria) | 1,19 | 0,5 % | el · producto · llegó · rápido · pero … |
| sílabas por reglas (tu idea) | 2,00 | 0,4 % | el · pro duc to · lle gó · rá pi do … |
| palabras enteras | 1,14 | 5,4 % | el · producto · llegó · rápido · pero … |

- Tal cual: la recompensa final es una «calidad» al azar, así que no aprende nada útil; con letras o prefijos que no vio, elige al azar.
- Arreglo: cada token cuesta −0,1 y −1 si no está entre los 8 000 más usados de su propia tokenización; ante algo nuevo sigue agregando letras mientras el token pueda llegar a ser uno conocido (como WordPiece).
- Falta medir cuánto acierta un clasificador con cada uno: `--tokenizadores` en Colab (cuaderno 11) ya los incluye.

## 3. Difusión de texto (`--difusion`)

Reseñas reales en español: 5 000 para entrenar y 500 de prueba, cortadas en 32 tokens (BPE de 4 000). Mismo Transformer de 2 capas para todos.

| Modelo | Rellena 15 % tapado | Rellena 50 % tapado | Perplejidad (menos es mejor) | Parámetros |
|---|---|---|---|---|
| difusión del cuaderno (tal cual) | 0,5 % ± 0,2 % | 0,4 % ± 0,1 % | — | 1 435 424 |
| **tu difusión con máscaras** (celda 127 del cuaderno completo) | 3,0 % ± 1,1 % | 3,0 % ± 0,5 % | — | 1 435 424 |
| difusión con máscaras (arreglo, como MDLM/LLaDA) | 3,4 % ± 0,5 % | 2,9 % ± 0,3 % | 803 ± 19 | 1 435 424 |
| Transformer autorregresivo (como GPT) | — | — | 263 ± 1 | 1 435 424 |

La perplejidad de la difusión con máscaras es una cota superior (la real puede ser menor). La difusión del cuaderno no define una probabilidad, así que no tiene perplejidad.

**Tu otra versión** (recuperada del cuaderno completo de Drive) ya tapaba palabras con [MASK], la misma idea que MDLM y LLaDA, y **rellena huecos igual de bien que la versión de la industria** (3,0 % contra 3,4 % y 2,9 %: empate dentro del ruido). Su punto débil es generar: elige siempre la palabra más probable y se queda repitiendo («que de que de…»). Con muestreo al azar, como MDLM, generaría palabras variadas. Su pérdida tampoco tiene la forma de MDLM, así que no da una perplejidad comparable.

## Ejemplos generados (semilla 0)

**difusión del cuaderno (tal cual)**:
- «sirve sirve sirve sirve sirve sirve sirve sirve sirve sirve sirve sirve sirve sirve sirve sirve sirve sirve sirve»
- «sirve sirve sirve sirve sirve sirve sirve sirve sirve sirve sirve sirve sirve sirve sirve sirve sirve sirve sirve»
- «sirve sirve sirve sirve sirve sirve sirve sirve sirve sirve sirve sirve sirve sirve sirve sirve sirve sirve sirve»

**tu difusión con máscaras (celda 127)**:
- «no que que que de que de de que que que de de de que que que que que» (las 3 muestras iguales: elige siempre la más probable)

**difusión con máscaras (arreglo, como MDLM/LLaDA)**:
- «sin pend" ni bateriaco, embalaje al lo dos a calidad en y unador se estetro»
- «olu tela tiposono en muy tresque viajar.e caj quec no tienes y ja»
- «con la con el... ins igual porque se de en las haca la juego li ordenador genial,»

**Transformer autorregresivo (como GPT)**:
- «ok en la comida al interior vienemas no llegado una cámara son al vendedor estado y noc»
- «la parte que apenas se cortata funcionaba nia, por demasiadod de unse el ar cumple»
- «perfecto en usarlo que tengo seri pungon que desce lleva mucha cadenatoerquina igual lo»
