# Informes del banco de evaluación

Un informe por tema, ordenados por cuaderno. Cada uno tiene la conclusión arriba y las tablas completas abajo (media ± desviación entre semillas, en datos de prueba que la red no vio al entrenar). Lo que recomiendo usar está resumido en [`RECOMENDACIONES.md`](../../../RECOMENDACIONES.md).

🟢 prometedora · ⚪ neutra (empata) · 🔴 no está lista

| Informe | Cuaderno | Qué se midió | En una línea |
|---|---|---|---|
| [`01-generativos-y-refuerzo.md`](01-generativos-y-refuerzo.md) | 01 texto | Agente PPO que predice palabras, tokenizador con refuerzo, difusión de texto | 🟢 tu difusión con máscaras rellena huecos como la de la industria; 🔴 PPO y tokenizador con refuerzo |
| [`01-tokenizadores-segmentadores-corrector.md`](01-tokenizadores-segmentadores-corrector.md) | 01 texto | Sílabas contra BPE, Unigram y GPT-4o; segmentadores de sílabas; corrector GNN | 🟢 sílabas le ganan a BPE; 🟢 BiLSTM de sílabas 99,2 %; ⚪ corrector híbrido empata con el diccionario |
| [`02-primera-tanda.md`](02-primera-tanda.md) | 02 capas | PReLU adaptativa, IDA, SMA, RMB, HMN, seno… | ⚪ casi todo empata; 🔴 HMN y RMB |
| [`02-memorias-e-industria.md`](02-memorias-e-industria.md) | 02 capas | Capas con memoria, inicializadores Perlin y Lorenz, topologías densas, contra GELU, SiLU, Mish… | ⚪ empatan; 🔴 memoria de errores y LR por neurona |
| [`02-neuronas-especiales.md`](02-neuronas-especiales.md) | 02 capas | Polinómica, resonante, DMNA, GMO, CDM, CCN, contra Dropout y BatchNorm | 🔴 polinómica, resonante, HiperReLU |
| [`02-dendritica-y-otras.md`](02-dendritica-y-otras.md) | 02 capas | Dendrítica, synaptic tagging, estado interno, ruteo, CNN de entrada densa | ⚪ empatan en tareas pequeñas |
| [`03-neuronas-alternativas.md`](03-neuronas-alternativas.md) | 03 | Neuronas KNN y redes genéticas (tal cual, arreglada, invertida) | 🔴 genéticas originales; ⚪ invertida |
| [`03-clasicos.md`](03-clasicos.md) | 03 | Logística, KNN, SVM, árbol, Random Forest, Gradient Boosting contra la red | Gradient Boosting y SVM ganan en varias tareas |
| [`04-decorrelacion-y-papelito.md`](04-decorrelacion-y-papelito.md) | 04 visión | CNN auto-decorrelada, papelito preguntón, celda RCF | ⚪ CNN decorrelada; 🔴 RCF |
| [`04-video.md`](04-video.md) | 04 visión | Tripletas y FrameInterpolator contra una U-Net | 🟢 tripletas con flujo: empatan con la U-Net con 19 veces menos parámetros |
| [`05-atencion.md`](05-atencion.md) | 05 atención | Transformer con máscara, AttentiveLinear, atención local | ⚪ máscara; 🔴 AttentiveLinear |
| [`06-cerebro-modular.md`](06-cerebro-modular.md) | 06 refuerzo | RlNeuron, votación, biomas, cerebro modular, tálamo | ⚪ empatan; la «victoria» del cerebro modular era por tamaño |
| [`06-refuerzo.md`](06-refuerzo.md) | 06 refuerzo | Entornos: escribir código, empujar cubos, ReroutingNetwork | 🔴 código; 🟢 cubos con guía (65 %) |
| [`07-anti-gradiente.md`](07-anti-gradiente.md) | 07 grafos | Capa anti-gradiente | 🔴 peor en 5 de 6 tareas |
| [`07-dreg-y-hcmg.md`](07-dreg-y-hcmg.md) | 07 generativos | D-ReG contra GRU/LSTM/Transformer; HCMG contra U-Net | ⚪ los dos; HCMG parecía ganar por una fuga de datos |
| [`08-votacion-stacking-y-caotica.md`](08-votacion-stacking-y-caotica.md) | 08 | Todas las piezas en la tarea caótica 20D; votación ponderada y stacking | 🟢 stacking (98,5 % en dígitos); 🟢 dendrítica en la caótica |
| [`reparadas.md`](reparadas.md) | 02, 03, 05, 07 | Las 8 piezas «no listas», arregladas, junto a su original; mejoras de la dendrítica y la base radial; anti-gradiente como DANN | Las 8 mejoran; 🟢 base radial selectiva pasa a Dropout en la caótica |

El efecto del tamaño de la red (por qué más parámetros no siempre es mejor) está en el cuaderno [`12_como_funciona_cada_pieza.ipynb`](../../../laboratorio-neuronal/12_como_funciona_cada_pieza.ipynb).

Para repetir cualquier informe, ver los comandos en [`laboratorio/banco/README.md`](../README.md) o en [`CLAUDE.md`](../../../CLAUDE.md).
