#!/usr/bin/env python3
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from functions import plot_results_per_class


samples = [10, 100, 1000]
percentiles = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]


def save_per_class_results_to_csv(coverage_results, accuracy_results, limits, sample_size, name):
    classes = sorted(coverage_results[limits[0]][sample_size].keys())
    data = []

    for class_label in classes:
        row = []
        for limit in limits:
            cov_val = coverage_results[limit][sample_size].get(class_label, 0)
            acc_val = accuracy_results[limit][sample_size].get(class_label, 0)

            def unpack(value):
                if isinstance(value, tuple):
                    return f"{value[0]:.2f} ± {value[1]:.2f}%"
                elif isinstance(value, list):
                    return f"{value[0]:.2f}%"
                else:
                    return f"{value:.2f}%"

            coverage = unpack(cov_val)
            accuracy = unpack(acc_val)

            row.append(f"Cov: {coverage}\nAcc: {accuracy}")
        data.append(row)

    columns = [f"Percentile_{p}" for p in limits]
    index = [("Total" if c == -1 else f"Class_{c}") for c in classes]
    df = pd.DataFrame(data, columns=columns, index=index)
    df.to_csv(f"/home/ev357/tcbench/src/fingerprinting/results/iterative_per_class/{name}.csv")


def save_confusion_matrix(preds, trues, sample, limit, name_prefix=""):
    y_true = []
    y_pred = []

    for label in trues[limit][sample]:
        y_true.extend(trues[limit][sample][label])
        y_pred.extend(preds[limit][sample][label])

    if not y_true:
        print(f"Skipping empty confusion matrix: sample={sample}, limit={limit}")
        return

    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, include_values=False, cmap='Blues')
    colorbar = disp.im_.colorbar
    colorbar.set_ticks([i / 10 for i in range(11)])
    colorbar.set_ticklabels([f"{int(i * 10)}%" for i in range(11)])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            ax.text(j, i, f"{int(round(value*100))}", ha='center', va='center', 
                    color='white' if value > 0.5 else 'black', fontsize=8)

    ax.set_title(f"{name_prefix} Sample {sample}, Percentile {limit}")

    out_dir = f"/home/ev357/tcbench/src/fingerprinting/conf_matrices/iterative_per_class/{name_prefix}"

    img_path = f"{out_dir}/sample{sample}_limit{limit}.png"
    plt.savefig(img_path)
    plt.close()


with open("/home/ev357/rds/hpc-work/results/iterative/embeddings_euclidean.pkl", "rb") as f:
    coverage_euc, accuracy_euc, preds_euc, trues_euc = pickle.load(f)

plot_results_per_class(coverage_euc, accuracy_euc, samples, percentiles, 'iterative_per_class_embeddings_splits_avg', percentile=True)
for s in samples:
    save_per_class_results_to_csv(coverage_euc, accuracy_euc, percentiles, s, f"embeddings_euclidean_sample{s}_avg")
for s in samples:
    for p in percentiles:
        save_confusion_matrix(preds_euc, trues_euc, s, p, name_prefix="Euclidean")


with open("/home/ev357/rds/hpc-work/results/iterative/embeddings_cosine.pkl", "rb") as f:
    coverage_cos, accuracy_cos, preds_cos, trues_cos = pickle.load(f)

plot_results_per_class(coverage_cos, accuracy_cos, samples, percentiles, 'iterative_per_class_embeddings_cosine_splits_avg', percentile=True)
for s in samples:
    save_per_class_results_to_csv(coverage_cos, accuracy_cos, percentiles, s, f"embeddings_cosine_sample{s}_splits_avg")
for s in samples:
    for p in percentiles:
        save_confusion_matrix(preds_cos, trues_cos, s, p, name_prefix="Cosine")