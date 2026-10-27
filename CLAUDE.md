# Notbook

Cuadernos de Google Colab ordenados: `laboratorio-neuronal/` (9 cuadernos de experimentos por tema) y `vulkan/` (portal Wi-Fi). Repo privado.

## Comandos

```bash
pip install nbformat
python .github/scripts/check_notebooks.py   # formato, sintaxis, llaves y resultados guardados
pip install -r laboratorio-neuronal/requirements.txt   # todas las librerías (pesado)
```

## Reglas del dueño (aplican a todo cambio)

- **Librerías**: si una dependencia está abandonada, sin mantenimiento, obsoleta o tiene una alternativa claramente más actual o potente, reemplázala sin pedir permiso. Explica el cambio y el motivo en el PR.
- **Quitar lo que no se usa**: archivos y dependencias que ningún código importa se eliminan. Menos código = menos superficie de ataque.
- **Nunca subir secretos**: ni llaves de API, ni contraseñas, ni archivos `.env`. Usa variables de entorno y `.env.example`.
- **Todo por pull request** hacia `main`, con las pruebas en verde antes de fusionar.
- **Probar antes de subir**: ejecuta las pruebas del repo; si cambia la interfaz, pruébala en un navegador.
- **Idioma**: README, PRs y comentarios para el usuario en español, claros y sin jerga.
- **Dependabot**: las actualizaciones menores se fusionan solas si las pruebas pasan; las mayores se revisan a mano.
- **Cuadernos**: guárdalos sin resultados; usa `torch.load(..., weights_only=True)`; nada de llaves en celdas (usa los Secretos de Colab).
- Al añadir una librería nueva a un cuaderno, agrégala también a `laboratorio-neuronal/requirements.txt` para que Dependabot la vigile.
