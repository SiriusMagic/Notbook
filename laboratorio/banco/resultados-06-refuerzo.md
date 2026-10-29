# Cuaderno 06: entornos de refuerzo

Código: `laboratorio/banco/refuerzo_06.py` (`python -m laboratorio.banco --refuerzo`; necesita `stable-baselines3`).

## Conclusión

- **Escribir código con PPO no funciona tal como está.** El objetivo del cuaderno («hola mundo») es imposible: el espacio no está entre los caracteres. Con un objetivo posible («hola», 10 acciones) PPO tampoco lo logra, ni con 300 000 pasos (lo que usaba el cuaderno): la recompensa grande solo llega cuando la línea completa está bien, y encontrarla probando al azar es casi imposible (1 entre 45¹⁰). Para generar código, la industria parte de un modelo que ya sabe escribir (aprendido de ejemplos) y después lo ajusta con refuerzo.
- **Empujar el cubo entre dos sí se aprende, con una guía.** Tal cual, PPO no llega nunca (0 %): el agente solo recibe señal cuando el cubo ya se movió, y para moverlo los dos cuerpos tienen que empujar a la vez. Con una recompensa pequeña por acercarse al cubo (*reward shaping*, lo que hace la industria) llega en el **60-70 %** de los episodios con 300 000 pasos.
- **ReroutingNetwork: REINFORCE tal cual es inestable.** Una semilla llega al óptimo (35,7), otra a 20,3 y otra se derrumba (1,1). Con la versión estándar (cada acción se juzga por lo que viene después, menos el promedio) las tres llegan al óptimo (35,6 de 35,8).


Cada agente se prueba en 30 episodios nuevos por semilla. «Éxito» = llegó a la meta.

## 1. Escribir código con PPO (50 000 pasos de entrenamiento, 2 semillas)

| Objetivo | Agente al azar | PPO (ajustes del cuaderno) |
|---|---|---|
| hola mundo (tal cual: sin espacio, imposible) | 0% | 0% ± 0% |
| hola (arreglo) | 0% | 0% ± 0% |
| hola, con 300 000 pasos (lo del cuaderno) | 0% | 0% ± 0% |

## 2. Empujar el cubo entre dos (cuadrícula 10×10, máximo 200 pasos, 2 semillas)

| Agente | Éxito | Pasos por episodio |
|---|---|---|
| al azar | 0% | 200 |
| PPO tal cual (100 000 pasos) | 0% ± 0% | 200 |
| PPO con posiciones normalizadas (100 000 pasos) | 0% ± 0% | 200 |
| PPO con posiciones normalizadas y guía hacia el cubo (300 000 pasos) | 65% ± 5% | 91 |

## 3. ReroutingNetwork (5 estados, 20 pasos por episodio, 3 semillas)

| Política | Recompensa por episodio |
|---|---|
| la mejor posible (calculada exacta) | 35,8 |
| REINFORCE tal cual (retorno total, sin línea base) | 19,0 ± 14,2 |
| REINFORCE estándar (recompensa futura menos el promedio) | 35,6 ± 0,0 |
| al azar | 18,1 |
