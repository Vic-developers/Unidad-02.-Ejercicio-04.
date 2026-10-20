# Model Card: clasificador Fashion-MNIST

## Estado de esta entrega

Los dos modelos se entrenaron aquí con el comando del README. Los resultados de esta corrida están en `reports/cv_metrics.json`; el notebook ejecutado incorpora tablas y figuras. El hardware y la semilla limitan la interpretación.

## Uso previsto

Uso educativo para comparar un clasificador denso y una CNN pequeña sobre Fashion-MNIST. No está destinado a decisiones automatizadas ni a clasificar ropa en fotografías del mundo real.

## Datos y partición

Fashion-MNIST, distribuido por `tf.keras.datasets.fashion_mnist`: 60,000 imágenes de entrenamiento originales y 10,000 de prueba oficial, en escala de grises de 28 × 28 píxeles y 10 clases. El conjunto de entrenamiento se divide de forma estratificada 90/10 usando semilla 42; el test oficial queda reservado hasta la evaluación final. Los hashes, conteos y rangos se guardan en `reports/data_audit.json` al ejecutar.

## Preprocesamiento

Conversión a `float32`, división por 255 y canal explícito para obtener forma `(N, 28, 28, 1)`. Las etiquetas son enteros de 0 a 9.

## Modelos

- Baseline: `Flatten → Dense(64, ReLU) → Dense(10, softmax)`.
- CNN: `Conv2D(32) → MaxPooling → Conv2D(64) → MaxPooling → GlobalAveragePooling → Dropout(0.25) → Dense(10, softmax)`.

Ambos usan Adam, entropía cruzada categórica dispersa, lote 128 por defecto, hasta 8 épocas y early stopping sobre `val_loss` restaurando el mejor checkpoint.

La comparación para elegir un modelo usa F1 macro de validación. Las métricas de test se reportan después y no se usan para seleccionar el ganador.

## Desempeño en test

| Modelo | Accuracy | F1 macro | Parámetros | Archivo |
| --- | ---: | ---: | ---: | ---: |
| Baseline denso | 0.8618 | 0.8605 | 50,890 | 619.1 KiB |
| CNN | 0.7891 | 0.7846 | 19,466 | 262.1 KiB |

La CNN redujo parámetros 61.7% y bytes 57.7%, pero tuvo F1 macro 0.0759 menor. La selección usa F1 macro de validación: 0.8769 para baseline y 0.7980 para CNN. La accuracy/F1 de test son evaluación final, no criterio de selección.

| Clase | F1 baseline | F1 CNN | Recall CNN |
| --- | ---: | ---: | ---: |
| T-shirt/top | 0.812 | 0.734 | 0.803 |
| Trouser | 0.974 | 0.954 | 0.929 |
| Pullover | 0.765 | 0.689 | 0.721 |
| Dress | 0.842 | 0.784 | 0.857 |
| Coat | 0.752 | 0.629 | 0.585 |
| Sandal | 0.955 | 0.917 | 0.877 |
| Shirt | 0.659 | 0.405 | 0.352 |
| Sneaker | 0.938 | 0.895 | 0.928 |
| Bag | 0.959 | 0.933 | 0.920 |
| Ankle boot | 0.950 | 0.906 | 0.919 |

## Costo computacional

Entorno: Python 3.12.13, TensorFlow 2.21.0, Windows 11, CPU Intel de 12 procesadores lógicos; no se detectó GPU. Train: 10.1 s baseline y 59.3 s CNN (5.88×). Latencia por imagen: 0.295 ± 0.027 ms baseline y 0.608 ± 0.086 ms CNN (2.06×), medida después de calentamiento sobre el mismo lote y cinco repeticiones. Los tiempos varían según hardware y backend.

## Limitaciones y riesgos

Las imágenes son pequeñas, centradas, monocromas y pertenecen a un benchmark didáctico. El resultado no demuestra rendimiento en fotos reales, otros catálogos, OCR, medicina, vigilancia ni contextos de decisión sobre personas. Un solo split y una sola semilla no estiman incertidumbre. Las clases visualmente similares pueden confundirse. Se requiere supervisión humana si se usa como demostración.

## Supervisión y mantenimiento

Inspeccionar errores y métricas por clase, fijar versiones y semillas, y repetir evaluación con varias semillas antes de cualquier conclusión comparativa. No desplegar fuera del uso educativo descrito sin datos representativos y evaluación específica.
