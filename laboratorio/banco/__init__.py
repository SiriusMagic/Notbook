"""Banco de evaluación: compara neuronas, activaciones y capas de investigación
contra una red estándar, con las mismas tareas y varias semillas (ver README.md)."""

from .evaluar import EVALUACIONES, comparar, correr, informe
from .piezas import PIEZAS
from .tareas import TAREAS, cargar_tarea

__all__ = ["EVALUACIONES", "PIEZAS", "TAREAS", "cargar_tarea", "comparar", "correr", "informe"]
