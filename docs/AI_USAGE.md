# Declaración de uso de IA

## Herramientas

Se utilizó ChatGPT/Codex como asistente para organizar el proyecto, proponer y revisar código, redactar documentación, preparar el notebook y generar el informe PDF. El estudiante debe revisar y poder explicar cada componente antes de presentarlo.

## Prompts relevantes de esta conversación

- «Necesito uno parecido a esto, que cumpla con todo esto», acompañado de los criterios del Ejercicio 04.
- Las instrucciones de visión computacional con Fashion-MNIST para comparar baseline denso y CNN, calcular métricas/costos, examinar errores y completar Model Card.
- «Agrégale esta hoja de presentación al documento PDF», con el formato académico solicitado, seguido de la instrucción para retirar el título, subtítulo y línea de autor que aparecían en la página interior.

El estudiante debe añadir aquí cualquier herramienta o prompt usado fuera de esta conversación que haya influido en la entrega.

## Partes verificadas

- Notebook con celdas ejecutadas y resultados guardados.
- Conteos, formas, rangos, distribución y hashes de las particiones en `reports/data_audit.json`.
- Métricas agregadas y por clase, matrices, curvas, errores visuales y mediciones de costo guardadas en `reports/`.
- Contratos de datos y predicción; en la ejecución registrada, `pytest` terminó con 3 pruebas aprobadas.
- PDF revisado mediante renderizado de sus páginas; el ZIP contiene el informe y los artefactos del proyecto.

## Correcciones realizadas

- Se eligió el modelo usando F1 macro de validación y se reservó test para la evaluación final.
- Se reportaron las mediciones de la ejecución local, en lugar de conservar cifras no verificadas del informe de referencia.
- El informe se regeneró con portada centrada y después se retiraron el título, subtítulo y atribución repetidos en su página interior.

## Procedencia y responsabilidad

No se recibió el ZIP del proyecto base de visión ni el notebook original de LAB07. Por ello, este repositorio contiene una ejecución reproducible de Fashion-MNIST creada para los requisitos proporcionados; no se afirma que integre o verifique los archivos originales de LAB07. El estudiante debe cotejarla con sus evidencias de laboratorio, agregar cualquier material requerido por el docente y confirmar que puede explicar los datos, código, referencias y conclusiones.
