"""Cuaderno 15 (banco de dispersión dinámica) y cuaderno 16 (superred con memoria portátil).

Cuaderno 15: un MLP con máscaras de conexiones y ranuras de neuronas, y los métodos que lo cambian mientras aprende
(``controlador``: MLP fijos, poda por magnitud, crecimiento programado, SET, RigL). RNRA-v0, que intentaba reasignar
conexiones con utilidad, redundancia y demanda por región, no superó a RigL ni al azar y se quitó (commit ``fd6e8f3``;
informe en ``laboratorio/banco/informes/15-reorganizacion-autonoma.md``).

- ``modelo``, ``operaciones``, ``controlador``, ``experimentos``: el banco del cuaderno 15.
- ``superred``, ``memoria_multiescala``, ``adaptador_dispositivo``, ``experimentos_m``: el cuaderno 16 (RNRA-M).
"""
