# Ejercicio 04: clasificador visual con CNN y Model Card

Proyecto reproducible para comparar un baseline denso y una CNN pequeña en Fashion-MNIST, con una única partición, auditoría, métricas globales y por clase, costo computacional, análisis visual de errores, pruebas y Model Card. El notebook de `notebooks/ejercicio_04_fashion_mnist.ipynb` recorre la ejecución.

## Requisitos

Se recomienda Python 3.12 y `uv`. Fashion-MNIST se descarga desde Keras la primera vez; no se incluye el dataset ni se debe subir `.venv`.

```powershell
uv sync --extra cpu --extra dev --extra pdf
uv run python scripts/check_runtime.py
uv run python scripts/train_cv.py --epochs 8
uv run pytest -q
```

Para GPU, use Linux/WSL2 con NVIDIA ya configurada: `uv sync --extra gpu`. En Colab, cargue el repositorio y ejecute `%pip install -r requirements-colab.txt`; después corra el notebook de arriba abajo. En Windows el comando CPU es el camino recomendado.

## Resultados y evidencia

La ejecución crea `reports/data_audit.json`, `reports/cv_metrics.json`, `reports/class_metrics.csv`, `reports/classification_reports.txt`, `reports/training_curves.png`, `reports/confusion_cnn.png`, `reports/cnn_errors.png` y los modelos `models/best_dense.keras` y `models/best_cnn.keras`. Este paquete incluye una ejecución real de esos artefactos en CPU; los tiempos corresponden al hardware informado en la auditoría. El test oficial de 10,000 imágenes se usa solo para evaluación final. Para actualizar métricas y figuras, vuelva a ejecutar el entrenamiento; genere el PDF con `uv run python scripts/build_report_pdf.py`. Repositorio de entrega: https://github.com/Vic-developers/Unidad-02.-Ejercicio-04..git.

## Decisiones reproducibles

- Semilla 42; partición estratificada del train oficial en 54,000/6,000.
- Normalización `[0, 1]` y forma `(N, 28, 28, 1)`.
- Mismos datos, batch, optimizador, criterio de parada y límite de épocas para ambos modelos.
- Early stopping monitoriza únicamente validación. El test reservado no participa en selección.
- La elección del modelo se basa en F1 macro de validación; las puntuaciones de test son evaluación final.
- Latencia medida en el mismo lote con calentamiento y cinco repeticiones; se reportan promedio y desviación.

## Estructura

`src/` contiene las funciones reutilizables; `scripts/` los comandos reproducibles; `tests/` los contratos; `notebooks/` la narrativa ejecutable; `reports/` resultados regenerables; `MODEL_CARD.md` límites y uso previsto.

Use `docs/error_analysis_guide.md` para redactar comentarios sobre casos concretos después de revisar los errores generados.

## Uso responsable de IA

Consulte `docs/AI_USAGE.md`. Las métricas deben verificarse ejecutando el proyecto. El estudiante conserva responsabilidad sobre código, datos, resultados y conclusiones.

## Referencias

- Fashion-MNIST, Zalando Research: https://github.com/zalandoresearch/fashion-mnist
- API de carga de Keras: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist/load_data
