#!/usr/bin/env python3
import numpy as np
import pandas as pd
from functions import (extract_results, 
                       extract_results_splits, 
                       plot_results, 
                       plot_results_splits)

samples = [10, 100, 1000]

distances_baseline = [0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3]
distances_embeddings = [0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
similarities = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 
                0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999]

coverage_results_baseline, accuracy_results_baseline, _ = extract_results('baseline', 'euclidean', 'distance', samples, distances_baseline)
plot_results(coverage_results_baseline, accuracy_results_baseline, samples, distances_baseline, "manual_all_classes_baseline")

coverage_results_embeddings, accuracy_results_embeddings, _ = extract_results('embeddings', 'euclidean', 'distance', samples, distances_embeddings)
plot_results(coverage_results_embeddings, accuracy_results_embeddings, samples, distances_embeddings, "manual_all_classes_embeddings")

coverage_results_embeddings_cosine, accuracy_results_embeddings_cosine, _ = extract_results('embeddings', 'cosine', 'similarity', samples, similarities)
plot_results(coverage_results_embeddings_cosine, accuracy_results_embeddings_cosine, samples, similarities, "manual_all_classes_embeddings_cosine", reverse=True)

def average_results_splits(coverage_results, accuracy_results, samples, limits):
    avg_coverage = {sample_size: {} for sample_size in samples}
    avg_accuracy = {sample_size: {} for sample_size in samples}
    for sample_size in samples:
        for limit in limits:
            avg_coverage[sample_size][limit] = np.mean(coverage_results[sample_size][limit])
            avg_accuracy[sample_size][limit] = np.mean(accuracy_results[sample_size][limit])
    return avg_coverage, avg_accuracy

def save_results_to_csv(avg_coverage, avg_accuracy, limits, metric, sample_sizes, name):
    data = []
    for sample_size in sample_sizes:
        row = []
        for limit in limits:
            coverage = avg_coverage[sample_size][limit]
            accuracy = avg_accuracy[sample_size][limit]
            row.append(f"Cov: {coverage:.2f}%\nAcc: {accuracy:.2f}%")
        data.append(row)

    if metric == "distance":
        columns = [f"Distance_{d}" for d in limits]
    elif metric == "similarity":
        columns = [f"Similarity_{s}" for s in limits]
    index = [f"Sample_{s}" for s in sample_sizes]
    df = pd.DataFrame(data, columns=columns, index=index)
    df.to_csv(f"/home/ev357/tcbench/src/fingerprinting/results/{name}.csv")


coverage_results_baseline_splits, accuracy_results_baseline_splits = extract_results_splits('baseline', 'euclidean', 'distance', samples, distances_baseline)
plot_results_splits(coverage_results_baseline_splits, accuracy_results_baseline_splits, samples, distances_baseline, "manual_all_classes_baseline_splits")
coverage_results_baseline_splits_avg, accuracy_results_baseline_splits_avg = average_results_splits(coverage_results_baseline_splits, accuracy_results_baseline_splits, samples, distances_baseline)
save_results_to_csv(coverage_results_baseline_splits_avg, accuracy_results_baseline_splits_avg, distances_baseline, 'distance', samples, 'manual_all_classes_baseline_splits_avg')

coverage_results_embeddings_splits, accuracy_results_embeddings_splits = extract_results_splits('embeddings', 'euclidean', 'distance', samples, distances_embeddings)
plot_results_splits(coverage_results_embeddings_splits, accuracy_results_embeddings_splits, samples, distances_embeddings, "manual_all_classes_embeddings_splits")
coverage_results_embeddings_splits_avg, accuracy_results_embeddings_splits_avg = average_results_splits(coverage_results_embeddings_splits, accuracy_results_embeddings_splits, samples, distances_embeddings)
save_results_to_csv(coverage_results_embeddings_splits_avg, accuracy_results_embeddings_splits_avg, distances_embeddings, 'distance', samples, 'manual_all_classes_embeddings_splits_avg')

coverage_results_embeddings_cosine_splits, accuracy_results_embeddings_cosine_splits = extract_results_splits('embeddings', 'cosine', 'similarity', samples, similarities)
plot_results_splits(coverage_results_embeddings_cosine_splits, accuracy_results_embeddings_cosine_splits, samples, similarities, "manual_all_classes_embeddings_cosine_splits", reverse=True)
coverage_results_embeddings_cosine_splits_avg, accuracy_results_embeddings_cosine_splits_avg = average_results_splits(coverage_results_embeddings_cosine_splits, accuracy_results_embeddings_cosine_splits, samples, similarities)
save_results_to_csv(coverage_results_embeddings_cosine_splits_avg, accuracy_results_embeddings_cosine_splits_avg, similarities, 'similarity', samples, 'manual_all_classes_embeddings_cosine_splits_avg')