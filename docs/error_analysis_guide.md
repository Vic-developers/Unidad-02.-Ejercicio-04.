# Guía breve para comentar errores

Cuando se generen `reports/cnn_errors.png` y `reports/confusion_cnn.png`, selecciona al menos tres ejemplos distintos y registra en el informe:

| Campo | Qué observar |
| --- | --- |
| Clase real y predicha | Copia ambas etiquetas del gráfico o de `reports/class_metrics.csv`. |
| Rasgo visual | Describe forma, silueta o detalle visible en la imagen pequeña. |
| Ambigüedad | Explica por qué las dos clases podrían parecerse sin atribuir causas que no observaste. |
| Métrica relacionada | Contrasta el ejemplo con recall/F1 por clase y con la matriz. |
| Límite | Recuerda que Fashion-MNIST no representa fotografías reales ni otros dominios. |

No infieras causa a partir de la matriz sola. Las imágenes son de baja resolución y la interpretación de los ejemplos es cualitativa.
