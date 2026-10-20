# Índice de evidencias frente a la rúbrica

Esta guía enlaza cada criterio con los artefactos que permiten revisarlo. Los resultados corresponden a la ejecución incluida en el repositorio; no deben generalizarse fuera de Fashion-MNIST.

| Criterio | Evidencia en esta entrega | Qué comprobar |
| --- | --- | --- |
| Auditoría y partición reproducible (0.75) | `reports/data_audit.json`, `src/cv_project.py`, notebook ejecutado | 54,000 train, 6,000 validación, 10,000 test; clases balanceadas, semilla 42, hashes y rangos de píxeles. |
| Baseline y CNN funcional (1.25) | `src/cv_project.py`, `scripts/train_cv.py`, `models/`, notebook | Arquitecturas, preprocesamiento compartido y partición común; los modelos guardados permiten inferencia. |
| Métricas, curvas y errores por clase (1.25) | `reports/cv_metrics.json`, `reports/class_metrics.csv`, `reports/classification_reports.txt`, `reports/training_curves.png`, `reports/confusion_cnn.png`, `reports/cnn_errors.png` | Comparar métricas globales y por clase; inspeccionar la curva, la matriz y ejemplos concretos de error. |
| Costo, tamaño y decisión técnica (0.75) | `reports/cv_metrics.json`, PDF, `MODEL_CARD.md` | Contrastar parámetros, tamaño, duración de entrenamiento y latencia; la decisión registrada prioriza F1 macro de validación. |
| Model Card, pruebas, README y Git (1.00) | `MODEL_CARD.md`, `tests/test_contracts.py`, `README.md`, `uv.lock`, historial Git y este índice | Revisar límites de uso, contratos, pasos de ejecución y trazabilidad del repositorio. La ejecución registrada reporta 3 pruebas aprobadas. |

## Procedencia pendiente de cotejo

No se recibió el proyecto base de visión ni el notebook original de LAB07. Esta entrega reproduce el ejercicio con Fashion-MNIST y conserva su propia ejecución, pero no permite demostrar que integra evidencias previas individuales del estudiante. Antes de someterla, cotejar este índice con los artefactos de LAB07 disponibles, incorporar los que el docente requiera y ejecutar las instrucciones del README en el entorno de entrega. No sustituir evidencia ausente con resultados supuestos.

## Decisión de esta corrida

El baseline denso obtuvo F1 macro de test 0.8605 frente a 0.7846 de la CNN. La CNN usa 61.7% menos parámetros y 57.7% menos espacio, pero en este hardware tardó 5.88 veces más en entrenar y tuvo 2.06 veces la latencia por imagen. Por F1 macro de validación se seleccionó el baseline. Estos tiempos son específicos de la corrida y el equipo documentados.
