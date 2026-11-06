"""Cuaderno 15 · Redes Neuronales de Reorganización Autónoma (RNRA-v0). **Sin validar.**

Una red con ranuras de neuronas y máscaras de conexiones que, mientras aprende, puede podar, activar neuronas de
reserva y reasignar conexiones hacia donde hace falta, con un presupuesto de conexiones controlado.

- ``modelo``: MLP con máscaras (pesos de tamaño máximo fijo; desactivar no ahorra tiempo real, solo cuenta conexiones).
- ``estadisticas``: utilidad, redundancia, plasticidad, estabilidad y demanda por región.
- ``operaciones``: poda, activación de neuronas de reserva y reorganización de conexiones.
- ``controlador``: el entrenador, el ciclo proponer → probar → aceptar o revertir y los comparadores (MLP fijos,
  poda por magnitud, crecimiento programado, SET, RigL, RNRA aleatoria y las ablaciones).
- ``experimentos``: E1-E5, semillas, estadísticas y el informe.
"""
