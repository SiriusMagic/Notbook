"""Revisa los cuadernos: formato válido, sintaxis de Python y llaves expuestas."""
import ast
import glob
import json
import re
import sys
import warnings

import nbformat

SECRET = re.compile(
    r"AIza[0-9A-Za-z_\-]{35}|hf_[A-Za-z0-9]{30,}|(?:ghp|gho|ghs|ghu)_[A-Za-z0-9]{30,}"
    r"|github_pat_[A-Za-z0-9_]{50,}|sk-(?:proj-)?[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}"
)
warnings.filterwarnings("ignore", category=SyntaxWarning)  # escapes en cadenas de los cuadernos
errors = []
paths = sorted(glob.glob("**/*.ipynb", recursive=True))
for path in paths:
    raw = open(path, encoding="utf-8").read()
    if SECRET.search(raw):
        errors.append(f"{path}: contiene algo que parece una llave de API")
    try:
        nbformat.validate(nbformat.reads(raw, as_version=4))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{path}: formato inválido ({exc.__class__.__name__})")
        continue
    for i, cell in enumerate(json.loads(raw)["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        # Las líneas de Colab (!pip, %magic) no son Python.
        code = "\n".join("pass" if l.lstrip().startswith(("!", "%", "pip ")) else l for l in src.splitlines())
        try:
            ast.parse(code)
        except SyntaxError as exc:
            print(f"aviso: {path} celda {i}: error de sintaxis previo ({exc.msg}, línea {exc.lineno})")
        if cell.get("outputs"):
            errors.append(f"{path} celda {i}: tiene resultados guardados; límpialos antes de subir")

print(f"{len(paths)} cuadernos revisados")
for e in errors:
    print("ERROR:", e)
sys.exit(1 if errors else 0)
