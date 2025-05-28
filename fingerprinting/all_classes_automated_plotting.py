#!/usr/bin/env python3
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from functions import (plot_results_splits)

samples = [10, 100, 1000]
percentiles = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]

def average_results_splits(coverage_results, accuracy_results, samples, limits):
    avg_coverage = {sample_size: {} for sample_size in samples}
    avg_accuracy = {sample_size: {} for sample_size in samples}
    for sample_size in samples:
        for limit in limits:
            avg_coverage[sample_size][limit] = np.mean(coverage_results[sample_size][limit])
            avg_accuracy[sample_size][limit] = np.mean(accuracy_results[sample_size][limit])
    return avg_coverage, avg_accuracy

def save_results_to_csv(avg_coverage, avg_accuracy, limits, sample_sizes, name):
    data = []
    for sample_size in sample_sizes:
        row = []
        for limit in limits:
            coverage = avg_coverage[sample_size][limit]
            accuracy = avg_accuracy[sample_size][limit]
            row.append(f"Cov: {coverage:.2f}%\nAcc: {accuracy:.2f}%")
        data.append(row)

    columns = [f"Percentile_{p}" for p in limits]
    index = [f"Sample_{s}" for s in sample_sizes]
    df = pd.DataFrame(data, columns=columns, index=index)
    df.to_csv(f"/home/ev357/tcbench/src/fingerprinting/results/automated_all_classes/{name}.csv")

def save_confusion_matrix(preds, trues, sample, limit, name_prefix=""): 
    y_true = trues[sample][limit]
    y_pred = preds[sample][limit]

    if not y_true:
        print(f"Skipping empty confusion matrix: sample={sample}, limit={limit}")
        return

    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, include_values=False, cmap='viridis')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            if value >= 0.05:
                ax.text(j, i, f"{value:.0%}", ha='center', va='center', color='white' if value > 0.5 else 'black', fontsize=8)

    ax.set_title(f"{name_prefix} Sample {sample}, Limit {limit}")

    out_dir = f"/home/ev357/tcbench/src/fingerprinting/conf_matrices/automated_all_classes/{name_prefix}"
    img_path = f"{out_dir}/sample{sample}_limit{limit}.png"
    plt.savefig(img_path)
    plt.close()


with open("/home/ev357/rds/hpc-work/results/all_classes/embeddings_euclidean.pkl", "rb") as f:
    cov_euc, acc_euc, preds_euc, trues_euc = pickle.load(f)

plot_results_splits(cov_euc, acc_euc, samples, percentiles, 'automated_all_classes_embeddings_euclidean_splits', percentile=True)
cov_avg_euc, acc_avg_euc = average_results_splits(cov_euc, acc_euc, samples, percentiles)
save_results_to_csv(cov_avg_euc, acc_avg_euc, percentiles, samples, 'embeddings_euclidean_avg')

for sample in samples:
    for limit in percentiles:
        save_confusion_matrix(preds_euc, trues_euc, sample, limit, name_prefix="Euclidean")


with open("/home/ev357/rds/hpc-work/results/all_classes/embeddings_cosine.pkl", "rb") as f:
    cov_cos, acc_cos, preds_cos, trues_cos = pickle.load(f)

plot_results_splits(cov_cos, acc_cos, samples, percentiles, 'automated_all_classes_embeddings_cosine_splits', percentile=True)
cov_avg_cos, acc_avg_cos = average_results_splits(cov_cos, acc_cos, samples, percentiles)
save_results_to_csv(cov_avg_cos, acc_avg_cos, percentiles, samples, 'embeddings_cosine_avg')

for sample in samples:
    for limit in percentiles:
        save_confusion_matrix(preds_cos, trues_cos, sample, limit, name_prefix="Cosine")
