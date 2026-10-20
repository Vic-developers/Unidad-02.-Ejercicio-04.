"""Build the concise report from the latest local run artifacts."""
import json
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak, Image)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Informe_Ejercicio_04.pdf"
metrics = json.loads((ROOT / "reports/cv_metrics.json").read_text(encoding="utf-8"))
audit = json.loads((ROOT / "reports/data_audit.json").read_text(encoding="utf-8"))
NAVY = colors.HexColor("#16324F")
GRAY = colors.HexColor("#F1F3F5")
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="CVTitle", parent=styles["Title"], fontName="Helvetica-Bold",
    fontSize=19, leading=23, textColor=NAVY, alignment=1, spaceAfter=6))
styles.add(ParagraphStyle(name="CVSub", parent=styles["Normal"], fontSize=9.5, leading=13,
    alignment=1, textColor=colors.HexColor("#425466"), spaceAfter=10))
styles.add(ParagraphStyle(name="CVH2", parent=styles["Heading2"], fontSize=12.5,
    leading=15, textColor=NAVY, spaceBefore=7, spaceAfter=4))
styles.add(ParagraphStyle(name="CVSmall", parent=styles["BodyText"], fontSize=8, leading=10))
styles.add(ParagraphStyle(name="CVNote", parent=styles["BodyText"], fontSize=8.5, leading=11,
    backColor=colors.HexColor("#FFF3CD"), borderPadding=6, spaceBefore=4, spaceAfter=6))

def footer(canvas, doc):
    canvas.saveState(); canvas.setFont("Helvetica", 8); canvas.setFillColor(colors.grey)
    canvas.drawString(.62*inch, .4*inch, "Ejercicio 04 · Fashion-MNIST · resultados de ejecución local")
    canvas.drawRightString(7.9*inch, .4*inch, f"{doc.page}"); canvas.restoreState()

def global_table():
    rows = [["Medida", "Baseline denso", "CNN"]]
    dense, cnn = metrics["models"]["dense"], metrics["models"]["cnn"]
    rows.extend([
        ["Accuracy (test)", f"{dense['accuracy']:.4f}", f"{cnn['accuracy']:.4f}"],
        ["F1 macro (test)", f"{dense['f1_macro']:.4f}", f"{cnn['f1_macro']:.4f}"],
        ["F1 macro (validación)", f"{dense['validation_f1_macro']:.4f}", f"{cnn['validation_f1_macro']:.4f}"],
        ["Parámetros", f"{dense['parameters']:,}", f"{cnn['parameters']:,}"],
        ["Tamaño", f"{dense['model_size_kib']:.1f} KiB", f"{cnn['model_size_kib']:.1f} KiB"],
        ["Entrenamiento", f"{dense['train_seconds']:.1f} s", f"{cnn['train_seconds']:.1f} s"],
        ["Latencia por imagen", f"{dense['inference']['ms_per_image_mean']:.3f} ± {dense['inference']['ms_per_image_std']:.3f} ms",
         f"{cnn['inference']['ms_per_image_mean']:.3f} ± {cnn['inference']['ms_per_image_std']:.3f} ms"],
    ])
    tab = Table(rows, colWidths=[2.0*inch, 2.3*inch, 2.3*inch], repeatRows=1)
    tab.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),NAVY),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTNAME",(0,1),(0,-1),"Helvetica-Bold"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,GRAY]),("GRID",(0,0),(-1,-1),.4,colors.HexColor("#D9D9D9")),
        ("ALIGN",(1,1),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("FONTSIZE",(0,0),(-1,-1),8),("LEFTPADDING",(0,0),(-1,-1),6),
        ("RIGHTPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ])); return tab

def class_table():
    names = audit["class_names"]
    rows = [["Clase", "Baseline F1", "CNN F1", "CNN recall"]]
    for cls in names:
        d = metrics["models"]["dense"]["per_class"][cls]
        c = metrics["models"]["cnn"]["per_class"][cls]
        rows.append([cls, f"{d['f1']:.3f}", f"{c['f1']:.3f}", f"{c['recall']:.3f}"])
    tab = Table(rows, colWidths=[2.65*inch, 1.25*inch, 1.25*inch, 1.25*inch], repeatRows=1)
    tab.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),NAVY),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,GRAY]),
        ("GRID",(0,0),(-1,-1),.4,colors.HexColor("#D9D9D9")),("ALIGN",(1,1),(-1,-1),"CENTER"),
        ("FONTSIZE",(0,0),(-1,-1),8),("LEFTPADDING",(0,0),(-1,-1),6),
        ("RIGHTPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
    ])); return tab

def main():
    dense, cnn = metrics["models"]["dense"], metrics["models"]["cnn"]
    comp = metrics["comparison"]
    selected = comp["selected_model_by_validation_f1"]
    hard_cls = min(cnn["per_class"], key=lambda c: cnn["per_class"][c]["f1"])
    cm = np.asarray(cnn["confusion_matrix"]); np.fill_diagonal(cm, 0)
    i, j = np.unravel_index(np.argmax(cm), cm.shape)
    names = audit["class_names"]
    story = [
        Paragraph("Clasificador visual reproducible con Fashion-MNIST", styles["CVTitle"]),
        Paragraph("Baseline denso y CNN · desempeño, costo, errores y Model Card", styles["CVSub"]),
        Paragraph("Universidad Autónoma de Santo Domingo · Ciencia de Datos II · Fraimel Trinidad · Prof. Edwin Ramón José Nolasco", styles["CVSmall"]),
        Paragraph("Ejecución realizada en este proyecto. La decisión del modelo se tomó con F1 macro de validación; test se reservó para la evaluación final. Repositorio de entrega: <link href='https://github.com/Vic-developers/Unidad-02.-Ejercicio-04..git' color='blue'>github.com/Vic-developers/Unidad-02.-Ejercicio-04..git</link>.", styles["CVNote"]),
        Paragraph("Resultado y decisión", styles["CVH2"]),
        Paragraph(f"Se selecciona <b>{selected}</b> por su F1 macro de validación. En test, el baseline obtuvo F1 macro {dense['f1_macro']:.4f} y accuracy {dense['accuracy']:.4f}; la CNN obtuvo {cnn['f1_macro']:.4f} y {cnn['accuracy']:.4f}. La CNN redujo parámetros {comp['parameter_reduction_cnn_percent']:.1f}% y tamaño {comp['size_reduction_cnn_percent']:.1f}%, pero tomó {comp['training_time_ratio_cnn_over_dense']:.2f}× más tiempo de entrenamiento y {comp['inference_latency_ratio_cnn_over_dense']:.2f}× más latencia por imagen en este equipo. Para esta corrida, la mejora de costo de almacenamiento no compensa el menor desempeño predictivo de la CNN.", styles["BodyText"]),
        Spacer(1,5), global_table(),
        Paragraph("Partición, entorno y método", styles["CVH2"]),
        Paragraph(f"Fashion-MNIST: train {audit['splits']['train']['n']:,}, validación {audit['splits']['validation']['n']:,} y test reservado {audit['splits']['test']['n']:,}; semilla {audit['seed']}. Preprocesamiento `float32`, escala [0,1], forma (28, 28, 1). Ambos modelos usan Adam, entropía cruzada dispersa, lote {metrics['batch_size']} y hasta {metrics['epochs_requested']} épocas con early stopping por val_loss. Runtime: Python {audit['python']}, TensorFlow {audit['tensorflow']}, {audit['platform']}, CPU (GPU visible: no).", styles["BodyText"]),
        PageBreak(),
        Paragraph("Desempeño por clase y errores visuales", styles["CVH2"]),
        Paragraph("F1 macro y recall CNN sobre test; soporte de 1,000 imágenes por clase. Los CSV del proyecto incluyen precisión, recall y F1 para ambos modelos.", styles["CVSmall"]),
        Spacer(1,4), class_table(), Spacer(1,5),
        Paragraph(f"La clase más difícil para la CNN fue <b>{hard_cls}</b> (F1 {cnn['per_class'][hard_cls]['f1']:.3f}, recall {cnn['per_class'][hard_cls]['recall']:.3f}). La mayor confusión fue {names[i]} → {names[j]} ({int(np.asarray(metrics['models']['cnn']['confusion_matrix'])[i,j])} casos). Las imágenes son ejemplos concretos del test.", styles["BodyText"]),
        Spacer(1,5),
    ]
    confusion = Image(str(ROOT / "reports/confusion_cnn.png"), width=3.22*inch, height=3.0*inch)
    errors = Image(str(ROOT / "reports/cnn_errors.png"), width=3.22*inch, height=3.0*inch)
    visual = Table([[confusion, errors]], colWidths=[3.3*inch,3.3*inch])
    visual.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE"),("ALIGN",(0,0),(-1,-1),"CENTER"),
                                ("LEFTPADDING",(0,0),(-1,-1),2),("RIGHTPADDING",(0,0),(-1,-1),2)]))
    story += [visual, Spacer(1,4),
        Paragraph("A la izquierda: matriz de confusión. A la derecha: 25 predicciones incorrectas seleccionadas por orden del conjunto test. Las similitudes visuales son interpretaciones cualitativas; la matriz no demuestra causalidad.", styles["CVSmall"]),
        Paragraph("Alcance, IA y repositorio", styles["CVH2"]),
        Paragraph("Fashion-MNIST es un benchmark educativo de imágenes pequeñas, centradas y monocromas. No demuestra preparación para fotos reales, OCR, medicina, vigilancia ni decisiones sobre personas. ChatGPT/Codex apoyó el scaffold, las pruebas y la redacción; el estudiante debe revisar los resultados y añadir cualquier herramienta o prompt adicional que haya utilizado. Repositorio de entrega: <link href='https://github.com/Vic-developers/Unidad-02.-Ejercicio-04..git' color='blue'>Unidad-02.-Ejercicio-04..git</link>.", styles["BodyText"])]
    doc = SimpleDocTemplate(str(OUT), pagesize=letter, rightMargin=.62*inch, leftMargin=.62*inch,
        topMargin=.5*inch, bottomMargin=.65*inch, title="Informe Ejercicio 04 Fashion-MNIST")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUT)

if __name__ == "__main__": main()
