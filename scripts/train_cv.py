"""Train both models and create reproducible reports and figures."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cv_project import (CLASS_NAMES, SEED, audit_splits, build_cnn, build_dense,
                        load_and_split, set_seed, write_json)


def inference_ms(model, x, repeats=5, batch_size=256):
    sample = x[:batch_size]
    model.predict(sample, verbose=0)  # warmup
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(sample, verbose=0)
        timings.append((time.perf_counter() - start) * 1000 / len(sample))
    return {"batch_size": len(sample), "repeats": repeats,
            "ms_per_image_mean": float(np.mean(timings)),
            "ms_per_image_std": float(np.std(timings, ddof=1))}


def plot_history(histories, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, h in histories.items():
        axes[0].plot(h.history["loss"], label=f"{name} train")
        axes[0].plot(h.history["val_loss"], linestyle="--", label=f"{name} val")
        axes[1].plot(h.history["accuracy"], label=f"{name} train")
        axes[1].plot(h.history["val_accuracy"], linestyle="--", label=f"{name} val")
    axes[0].set(title="Loss", xlabel="Época", ylabel="Loss")
    axes[1].set(title="Accuracy", xlabel="Época", ylabel="Accuracy")
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def plot_confusion(cm, out):
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(10), yticks=range(10), xticklabels=CLASS_NAMES,
           yticklabels=CLASS_NAMES, xlabel="Predicción", ylabel="Clase real",
           title="Matriz de confusión CNN (test)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(10):
        for j in range(10): ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def plot_errors(x, y, pred, out, limit=25):
    wrong = np.flatnonzero(pred != y)[:limit]
    cols = 5; rows = max(1, int(np.ceil(len(wrong) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.5 * rows), squeeze=False)
    for ax in axes.flat: ax.axis("off")
    for ax, ix in zip(axes.flat, wrong):
        ax.imshow(x[ix].squeeze(), cmap="gray")
        ax.set_title(f"Real: {CLASS_NAMES[y[ix]]}\nPred: {CLASS_NAMES[pred[ix]]}", fontsize=8)
        ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Errores CNN en test: {len(y) - int((pred == y).sum())} total; muestra de {len(wrong)}")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def evaluate(name, model, x, y, model_path, train_seconds):
    probs = model.predict(x, batch_size=512, verbose=0)
    pred = probs.argmax(axis=1)
    pr, rc, f1, support = precision_recall_fscore_support(y, pred, labels=range(10), zero_division=0)
    per_class = {CLASS_NAMES[i]: {"precision": float(pr[i]), "recall": float(rc[i]),
                                   "f1": float(f1[i]), "support": int(support[i])} for i in range(10)}
    size = model_path.stat().st_size
    return {
        "accuracy": float(accuracy_score(y, pred)), "f1_macro": float(f1_score(y, pred, average="macro")),
        "precision_macro": float(np.mean(pr)), "recall_macro": float(np.mean(rc)),
        "per_class": per_class, "parameters": int(model.count_params()),
        "model_size_bytes": int(size), "model_size_kib": float(size / 1024),
        "train_seconds": float(train_seconds), "inference": inference_ms(model, x),
        "classification_report": classification_report(y, pred, target_names=CLASS_NAMES,
                                                         output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, pred, labels=range(10)).tolist(),
    }, pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1: parser.error("epochs y batch-size deben ser positivos")
    out = args.output.resolve(); reports = out / "reports"; models_dir = out / "models"
    reports.mkdir(parents=True, exist_ok=True); models_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    splits = load_and_split(args.seed)
    audit = audit_splits(splits, args.seed)
    write_json(reports / "data_audit.json", audit)
    x_train, y_train = splits["train"]; x_val, y_val = splits["validation"]
    x_test, y_test = splits["test"]
    metrics = {"seed": args.seed, "epochs_requested": args.epochs,
               "batch_size": args.batch_size, "test_n": len(y_test), "models": {}}
    histories = {}
    predictions = {}
    for name, builder in [("dense", build_dense), ("cnn", build_cnn)]:
        set_seed(args.seed)
        model = builder()
        path = models_dir / ("best_" + name + ".keras")
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience,
                    restore_best_weights=True),
                     tf.keras.callbacks.ModelCheckpoint(path, monitor="val_loss", save_best_only=True)]
        t0 = time.perf_counter()
        h = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=args.epochs,
                      batch_size=args.batch_size, callbacks=callbacks, verbose=2)
        train_s = time.perf_counter() - t0
        histories[name] = h
        model = tf.keras.models.load_model(path)
        metrics["models"][name], pred = evaluate(name, model, x_test, y_test, path, train_s)
        predictions[name] = pred
        val_prob = model.predict(x_val, batch_size=512, verbose=0)
        metrics["models"][name]["validation_f1_macro"] = float(
            f1_score(y_val, val_prob.argmax(axis=1), average="macro"))
        if name == "cnn":
            plot_confusion(np.array(metrics["models"][name]["confusion_matrix"]), reports / "confusion_cnn.png")
            plot_errors(x_test, y_test, pred, reports / "cnn_errors.png")
        print(f"{name}: accuracy={metrics['models'][name]['accuracy']:.4f}, "
              f"F1 macro={metrics['models'][name]['f1_macro']:.4f}")
    plot_history(histories, reports / "training_curves.png")
    dense, cnn = metrics["models"]["dense"], metrics["models"]["cnn"]
    metrics["comparison"] = {
        "f1_macro_delta_cnn_minus_dense": cnn["f1_macro"] - dense["f1_macro"],
        "accuracy_delta_cnn_minus_dense": cnn["accuracy"] - dense["accuracy"],
        "parameter_reduction_cnn_percent": (1 - cnn["parameters"] / dense["parameters"]) * 100,
        "size_reduction_cnn_percent": (1 - cnn["model_size_bytes"] / dense["model_size_bytes"]) * 100,
        "training_time_ratio_cnn_over_dense": cnn["train_seconds"] / max(dense["train_seconds"], 1e-12),
        "inference_latency_ratio_cnn_over_dense": cnn["inference"]["ms_per_image_mean"] / max(dense["inference"]["ms_per_image_mean"], 1e-12),
        "selected_model_by_validation_f1": "dense" if dense["validation_f1_macro"] >= cnn["validation_f1_macro"] else "cnn",
    }
    write_json(reports / "cv_metrics.json", metrics)
    rows = []
    for name, result in metrics["models"].items():
        for cls, vals in result["per_class"].items(): rows.append({"model": name, "class": cls, **vals})
    pd.DataFrame(rows).to_csv(reports / "class_metrics.csv", index=False)
    (reports / "classification_reports.txt").write_text(
        "\n\n".join(f"{name.upper()}\n" + classification_report(
            y_test, predictions[name], target_names=CLASS_NAMES, zero_division=0)
            for name in predictions), encoding="utf-8")
    print(f"Reports: {reports}")


if __name__ == "__main__": main()
