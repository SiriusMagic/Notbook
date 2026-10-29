# Cuaderno 01: tokenizadores, segmentadores de sílabas y corrector

Medido en CPU con reseñas reales en español (`resenas_es`): 3 000 para entrenar y 1 000 de prueba que no se vieron. Código: `tokenizadores.py`, `segmentadores.py` y `correccion.py`.

## Conclusión

- **Tokenizadores: tus sílabas le ganan a BPE de la industria.** Al clasificar el puntaje de las reseñas (5 clases; al azar, 20 %) aciertan **40,0 % contra 35,8 %** con 3 000 reseñas (5 semillas, significativo) y **45,0 % contra 41,5 %** con 12 000 (3 semillas). Ganan en las 8 semillas. La razón: cortar más fino ayuda cuando hay pocos datos, porque las sílabas se repiten mucho más que las palabras o los trozos de BPE. Las letras sueltas ganan por un punto más, pero con secuencias 2,2 veces más largas (más lentas). Las sílabas son el punto medio. Probé dos mejoras que juntan sílabas para acortar (sílabas + BPE, y palabras frecuentes enteras): quedan entre BPE y las sílabas, así que juntar le quita la ventaja.
- **Segmentadores: tu BiLSTM de sílabas acierta 99,2 % de las palabras**, mucho más que pyphen (71,1 %, la librería que usan LibreOffice y TeX) y que la heurística original del cuaderno (59,4 %). Las reglas escritas a mano llegan a 100 % y son 80 veces más rápidas, así que la red solo vale la pena si se quiere aprender de ejemplos (otro idioma, nombres raros).
- **Corrector:** con solo letras cambiadas, tu GNN con conexiones residuales deja el texto en 11,0 % de letras mal (sin corregir: 12,6 %), y el diccionario estilo Norvig/SymSpell en 6,6 %. La mejora **híbrida** (el diccionario decide qué palabras están mal y la red elige por cuál cambiarlas) llega a 6,8 %: empata con el diccionario, pero **la red no le suma**. Con todos los tipos de error pasa lo mismo (10,2 % contra 10,1 %). Para que un corrector le gane al diccionario hace falta contexto de palabras, no de letras. Falta la BiLSTM y el Transformer en GPU (cuaderno 13).

## 1. Tokenizadores: clasificar reseñas con la misma red (CNN de texto)

Todos en minúsculas y con el mismo vocabulario (8 000 unidades); solo cambia cómo se corta el texto. 15 épocas. Se compara con BPE semilla por semilla (mismos datos y mismo inicio de la red). El Transformer y las posiciones jerárquicas van en el cuaderno 13 (GPU).

### 3 000 reseñas para entrenar, 1 000 de prueba, 5 semillas

| Tokenizador | Acierto | Le gana a BPE en… | Tokens por palabra | Desconocidas |
|---|---|---|---|---|
| letras sueltas | 41,9 ± 1,7 % | 5 de 5 semillas | 4,50 | 0,01 % |
| **sílabas por reglas** (tuya) | 40,0 ± 1,6 % | 5 de 5 semillas | 2,00 | 0,40 % |
| palabras frecuentes enteras + sílabas (mejora) | 38,0 ± 1,6 % | 4 de 5 semillas | 1,34 | 0,74 % |
| sílabas + BPE (mejora: junta sílabas frecuentes) | 37,6 ± 1,0 % | 5 de 5 semillas | 1,26 | 1,79 % |
| Unigram (industria: T5, Gemma) | 37,4 ± 1,7 % | 5 de 5 semillas | 1,24 | 0,24 % |
| BPE (industria: GPT-2, Llama) | 35,8 ± 1,6 % | — | 1,19 | 0,44 % |

Con 5 semillas la ventaja de tus sílabas sobre BPE es **significativa** (prueba t pareada corregida por comparaciones múltiples).

### 12 000 reseñas para entrenar, 2 000 de prueba, 3 semillas

| Tokenizador | Acierto | Le gana a BPE en… | Tokens por palabra | Desconocidas |
|---|---|---|---|---|
| letras sueltas | 45,6 ± 0,7 % | 3 de 3 semillas | 4,49 | 0,00 % |
| **sílabas por reglas** (tuya) | 45,0 ± 0,5 % | 3 de 3 semillas | 2,00 | 0,18 % |
| Unigram (industria: T5, Gemma) | 42,0 ± 1,5 % | 2 de 3 semillas | 1,22 | 0,01 % |
| BPE (industria: GPT-2, Llama) | 41,5 ± 1,0 % | — | 1,17 | 0,03 % |

Con 4 veces más datos todos suben y **la ventaja de las sílabas se mantiene** (+3,5 puntos, en las 3 semillas). Con solo 3 semillas la prueba no alcanza a declararla significativa.

### Primera medición (todos los tokenizadores, 2 semillas)

| Tokenizador | Acierto | Tokens por palabra | Desconocidas |
|---|---|---|---|
| letras sueltas | 40,6 ± 1,2 % | 4,50 | 0,01 % |
| con refuerzo, tal cual (tuya) | 40,3 ± 0,4 % | 3,67 | 0,20 % |
| sílabas por reglas (tuya) | 39,9 ± 1,7 % | 2,00 | 0,40 % |
| sílabas del cuaderno (tuya, tal cual) | 39,6 ± 0,1 % | 1,97 | 0,48 % |
| con refuerzo, arreglado (tuya) | 37,7 ± 0,1 % | 1,22 | 2,93 % |
| palabras enteras (word2vec, fastText) | 37,7 ± 2,1 % | 1,13 | 5,45 % |
| Unigram | 36,7 ± 2,3 % | 1,24 | 0,24 % |
| BPE | 35,2 ± 2,8 % | 1,19 | 0,44 % |
| GPT-4o (sin entrenar) | 34,7 ± 1,2 % | 1,28 | 2,65 % |

Menos tokens por palabra = secuencias más cortas = modelo más rápido. Pocas desconocidas = el vocabulario cubre bien textos nuevos.

## 2. Segmentadores de sílabas

Medidos con 128 palabras silabeadas a mano según la RAE (`silabas_referencia.py`). Las redes aprendieron de 15,328 palabras reales de reseñas etiquetadas con las reglas (2 semillas). Aviso: la lista de referencia y las reglas son del mismo autor.

| Segmentador | Palabra exacta | F1 de cortes | Palabras/s | Parámetros | Seg. entrenar |
|---|---|---|---|---|---|
| reglas | 100.0 % | 100.0 % | 190,990 | — (reglas) | — |
| bilstm | 99.2 % ± 0.0 | 99.7 % ± 0.0 | 2,300 | 51,425 | 15.4 |
| transformer | 98.8 % ± 1.7 | 99.5 % ± 0.7 | 2,321 | 70,785 | 72.5 |
| pyphen | 71.1 % | 88.0 % | 93,438 | — (reglas) | — |
| cuaderno | 59.4 % | 75.1 % | 341,448 | — (reglas) | — |

## Ejemplos de errores

- **reglas**: sin errores.
- **cuaderno** (52 errores): palabra: `pa-labr-a` (es `pa-la-bra`), otro: `otr-o` (es `o-tro`), libro: `libr-o` (es `li-bro`), hablar: `habl-ar` (es `ha-blar`), problema: `probl-e-ma` (es `pro-ble-ma`), tecla: `tecl-a` (es `te-cla`)
- **pyphen** (37 errores): otro: `otro` (es `o-tro`), abrazo: `abra-zo` (es `a-bra-zo`), aplauso: `aplau-so` (es `a-plau-so`), ofrecer: `ofre-cer` (es `o-fre-cer`), agradable: `agra-da-ble` (es `a-gra-da-ble`), pingüino: `pin-güino` (es `pin-güi-no`)
- **bilstm** (1 errores): aéreo: `aé-re-o` (es `a-é-re-o`)
- **transformer** (3 errores): pingüino: `pin-g-üi-no` (es `pin-güi-no`), vergüenza: `ver-g-üen-za` (es `ver-güen-za`), sonreír: `so-nre-ír` (es `son-re-ír`)

## 3. Corrector ortográfico

3 000 frases para entrenar, 500 de prueba. CER = % de letras mal después de corregir (menos es mejor).

| Corrector | Solo letras cambiadas | Todos los errores |
|---|---|---|
| Diccionario (Norvig/SymSpell, industria clásica) | **6,6 %** | **10,1 %** |
| **Híbrido: diccionario + tu GNN** (mejora, 3 semillas) | 6,8 % | 10,2 % |
| Tu GNN con conexiones residuales (arreglo, 3 semillas) | 11,0 % | 20,6 % |
| Sin corregir (piso) | 12,6 % | 16,6 % |
| Tu GNN tal cual, supervisada (2 semillas) | 43,6 % | — |
| Tu GNN tal cual, con refuerzo, como el cuaderno (2 semillas) | 94,1 % | — |

Con todos los errores (letras de más, de menos, traspuestas y espacios) tu GNN empeora el texto: corrige letra por letra y no puede mover letras de lugar.

Para repetirlo:

```bash
python -m laboratorio.banco --tokenizadores --tarea resenas_es --entreno 3000 --prueba 1000 --semillas 2 --epocas 15
python -m laboratorio.banco --segmentadores --semillas 2
python -m laboratorio.banco --correccion --entreno 3000 --prueba 500 --semillas 2
```
