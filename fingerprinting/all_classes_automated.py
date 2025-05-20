#!/usr/bin/env python3
import numpy as np
import pandas as pd
from functions import (extract_results_limits,
                       extract_results_limits_splits, 
                       plot_results,
                       plot_results_splits)

samples = [10, 100, 1000]
percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]

coverage_results_baseline, accuracy_results_baseline = extract_results_limits('baseline', 'euclidean', 'distance', samples, percentiles)
plot_results(coverage_results_baseline, accuracy_results_baseline, samples, percentiles, 'automated_all_classes_baseline', percentile=True)

coverage_results_embeddings, accuracy_results_embeddings = extract_results_limits('embeddings', 'euclidean', 'distance', samples, percentiles)
plot_results(coverage_results_embeddings, accuracy_results_embeddings, samples, percentiles, 'automated_all_classes_embeddings', percentile=True)

coverage_results_embeddings_cosine, accuracy_results_embeddings_cosine, = extract_results_limits('embeddings', 'cosine', 'similarity', samples, percentiles)
plot_results(coverage_results_embeddings_cosine, accuracy_results_embeddings_cosine, samples, percentiles, 'automated_all_classes_embeddings_cosine', percentile=True)


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


coverage_results_baseline_splits, accuracy_results_baseline_splits = extract_results_limits_splits('baseline', 'euclidean', 'distance', samples, percentiles)
plot_results_splits(coverage_results_baseline_splits, accuracy_results_baseline_splits, samples, percentiles, 'automated_all_classes_baseline_splits', percentile=True)
coverage_results_baseline_splits_avg, accuracy_results_baseline_splits_avg = average_results_splits(coverage_results_baseline_splits, accuracy_results_baseline_splits, samples, percentiles)
save_results_to_csv(coverage_results_baseline_splits_avg, accuracy_results_baseline_splits_avg, percentiles, 'distance', samples, 'automated_all_classes_baseline_splits_avg')

coverage_results_embeddings_splits, accuracy_results_embeddings_splits = extract_results_limits_splits('embeddings', 'euclidean', 'distance', samples, percentiles)
plot_results_splits(coverage_results_embeddings_splits, accuracy_results_embeddings_splits, samples, percentiles, 'automated_all_classes_embeddings_splits', percentile=True)
coverage_results_embeddings_splits_avg, accuracy_results_embeddings_splits_avg = average_results_splits(coverage_results_embeddings_splits, accuracy_results_embeddings_splits, samples, percentiles)
save_results_to_csv(coverage_results_embeddings_splits_avg, accuracy_results_embeddings_splits_avg, percentiles, 'distance', samples, 'automated_all_classes_embeddings_splits_avg')

coverage_results_embeddings_cosine_splits, accuracy_results_embeddings_cosine_splits = extract_results_limits_splits('embeddings', 'cosine', 'similarity', samples, percentiles)
plot_results_splits(coverage_results_embeddings_cosine_splits, accuracy_results_embeddings_cosine_splits, samples, percentiles, 'automated_all_classes_embeddings_cosine_splits', percentile=True)
coverage_results_embeddings_cosine_splits_avg, accuracy_results_embeddings_cosine_splits_avg = average_results_splits(coverage_results_embeddings_cosine_splits, accuracy_results_embeddings_cosine_splits, samples, percentiles)
save_results_to_csv(coverage_results_embeddings_cosine_splits_avg, accuracy_results_embeddings_cosine_splits_avg, percentiles, 'similarity', samples, 'automated_all_classes_embeddings_cosine_splits_avg')
