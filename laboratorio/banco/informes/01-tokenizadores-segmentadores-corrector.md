# Cuaderno 01: tokenizadores, segmentadores de sílabas y corrector

Medido en CPU con reseñas reales en español (`resenas_es`): 3 000 para entrenar y 1 000 de prueba que no se vieron. Código: `tokenizadores.py`, `segmentadores.py` y `correccion.py`.

## Conclusión

- **Tokenizadores: con pocos datos, cortar fino ayuda un poco.** Tus sílabas aciertan 39,9 % al clasificar el puntaje de las reseñas (5 clases; al azar, 20 %); BPE de la industria, 35,2 %. Las letras sueltas (40,6 %) y tu tokenizador con refuerzo tal cual (40,3 %, que corta casi letra por letra) quedan igual de arriba. Con 2 semillas **ninguna diferencia es significativa**: es una pista, no una victoria. Con muchos datos y modelos grandes, la industria usa BPE porque las secuencias son más cortas (1,2 tokens por palabra contra 2,0 de las sílabas).
- **Segmentadores: tu BiLSTM de sílabas acierta 99,2 % de las palabras**, mucho más que pyphen (71,1 %, la librería que usan LibreOffice y TeX) y que la heurística original del cuaderno (59,4 %). Las reglas escritas a mano llegan a 100 % y son 80 veces más rápidas, así que la red solo vale la pena si se quiere aprender de ejemplos (otro idioma, nombres raros).
- **Corrector (medición parcial):** con solo letras cambiadas, tu GNN con conexiones residuales deja el texto en 10,8 % de letras mal (sin corregir: 12,6 %). El diccionario estilo Norvig/SymSpell llega a 6,7 %. Sin residuales el GNN empeora el texto (43,6 %) y con refuerzo, como en el cuaderno, lo destruye (94,1 %). Falta la BiLSTM, el Transformer y el escenario con todos los errores: en CPU tardaba más de 2 horas, va en el cuaderno 13 (GPU).

## 1. Tokenizadores: clasificar reseñas con la misma red (CNN de texto)

Todos en minúsculas y con el mismo vocabulario (8 000 unidades); solo cambia cómo se corta el texto. 15 épocas, 2 semillas. El Transformer y las posiciones jerárquicas van en el cuaderno 13 (GPU).

| Tokenizador | Acierto al clasificar (2 semillas) | Tokens por palabra | Desconocidas |
|---|---|---|---|
| letras sueltas | 40,6 ± 1,2 % | 4,50 | 0,01 % |
| con refuerzo, tal cual (tuya) | 40,3 ± 0,4 % | 3,67 | 0,20 % |
| **sílabas por reglas** (tuya, corregida) | 39,9 ± 1,7 % | 2,00 | 0,40 % |
| sílabas del cuaderno (tuya, tal cual) | 39,6 ± 0,1 % | 1,97 | 0,48 % |
| con refuerzo, arreglado (tuya) | 37,7 ± 0,1 % | 1,22 | 2,93 % |
| palabras enteras (word2vec, fastText) | 37,7 ± 2,1 % | 1,13 | 5,45 % |
| Unigram (industria: T5, Llama 1) | 36,7 ± 2,3 % | 1,24 | 0,24 % |
| BPE (industria: GPT-2, RoBERTa) | 35,2 ± 2,8 % | 1,19 | 0,44 % |
| GPT-4o (industria, sin entrenar) | 34,7 ± 1,2 % | 1,28 | 2,65 % |

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

## 3. Corrector ortográfico (parcial, escenario «solo letras cambiadas»)

3 000 frases para entrenar, 500 de prueba, 2 semillas. CER = % de letras mal después de corregir (menos es mejor).

| Corrector | CER |
|---|---|
| Diccionario (Norvig/SymSpell, industria clásica) | 6,66 % |
| **Tu GNN con conexiones residuales** (arreglo) | **10,84 %** |
| Sin corregir (piso) | 12,56 % |
| Tu GNN tal cual, supervisada | 43,57 % |
| Tu GNN tal cual, con refuerzo (como el cuaderno) | 94,10 % |

Para repetirlo:

```bash
python -m laboratorio.banco --tokenizadores --tarea resenas_es --entreno 3000 --prueba 1000 --semillas 2 --epocas 15
python -m laboratorio.banco --segmentadores --semillas 2
python -m laboratorio.banco --correccion --entreno 3000 --prueba 500 --semillas 2
```
