# Guía para crear y usar entornos virtuales en Python

Esta guía explica cómo:
1. Crear un entorno virtual (venv)
2. Activarlo
3. Instalar dependencias desde un archivo `requirements.txt`

---

## ⭐ 1. Crear un entorno virtual (venv)

Asegúrate de tener Python 3.3+ instalado.

En la terminal, ejecuta:

```bash
python -m venv .venv
```
## ⭐ 2. Activar el entorno virtual

```bash
.venv\Scripts\activate
```
## ⭐ 3. Descargar las dependencias

```bash
pip install -r requirements.txt
```
## ⭐ 4. Iniciar el programa 

```bash
python kmeans_app.py
```

Prueba git push