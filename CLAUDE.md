# Notbook

Cuadernos de Google Colab ordenados: `laboratorio-neuronal/` (9 cuadernos de experimentos por tema), `laboratorio/` (piezas reutilizables en `.py` con pruebas) y `vulkan/` (portal Wi-Fi). Repo privado.

## Comandos

```bash
pip install nbformat
python .github/scripts/check_notebooks.py   # formato, sintaxis, llaves y resultados guardados
pip install -r laboratorio-neuronal/requirements.txt   # todas las librerías (pesado)
pip install -r laboratorio/requirements.txt && pytest   # pruebas de laboratorio/ (torch opcional)
python -m laboratorio.banco --todo --semillas 5          # banco de evaluación (~5 min en CPU) → resultados/informe.md
python -m laboratorio.banco --reales --guardar modelos     # con MNIST, CIFAR-10, reseñas… de Hugging Face (parada temprana activada)
python -m laboratorio.banco --competencia --rapido          # piezas contra CNN, TextCNN, Transformer, CNN 1D y LSTM (mejor con GPU)
python -m laboratorio.banco --tokenizadores                 # cuaderno 01: sílabas contra BPE, Unigram y GPT-4o
python -m laboratorio.banco --segmentadores                 # cuaderno 01: segmentadores de sílabas
python -m laboratorio.banco --correccion                    # cuaderno 01: corrección ortográfica (tu GNN)
python -m laboratorio.banco --clasicos                      # cuaderno 03: Random Forest, Gradient Boosting, SVM… contra la red
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
