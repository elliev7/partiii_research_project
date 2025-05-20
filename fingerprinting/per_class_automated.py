#!/usr/bin/env python3
import pandas as pd
from functions import (extract_results_limits_per_class,
                       extract_results_limits_per_class_splits_avg,
                       plot_results_per_class)

samples = [10, 100, 1000]
percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]

coverage_results_baseline_limits_per_class, accuracy_results_baseline_limits_per_class, missed_results_baseline_limits_per_class = extract_results_limits_per_class('baseline', 'euclidean', 'distance', samples, percentiles)
plot_results_per_class(coverage_results_baseline_limits_per_class, accuracy_results_baseline_limits_per_class, samples, percentiles, 'automated_per_class_baseline', percentile=True)

coverage_results_embeddings_limits_per_class, accuracy_results_embeddings_limits_per_class, missed_results_embeddings_limits_per_class = extract_results_limits_per_class('embeddings', 'euclidean', 'distance', samples, percentiles)
plot_results_per_class(coverage_results_embeddings_limits_per_class, accuracy_results_embeddings_limits_per_class, samples, percentiles, 'automated_per_class_embeddings', percentile=True)

coverage_results_embeddings_cosine_limits_per_class, accuracy_results_embeddings_cosine_limits_per_class, missed_results_embeddings_cosine_limits_per_class = extract_results_limits_per_class('embeddings', 'cosine', 'similarity', samples, percentiles)
plot_results_per_class(coverage_results_embeddings_cosine_limits_per_class, accuracy_results_embeddings_cosine_limits_per_class, samples, percentiles, 'automated_per_class_embeddings_cosine', percentile=True)

def save_per_class_results_to_csv(coverage_results, accuracy_results, missed_results, limits, sample_size, name):
    classes = sorted(coverage_results[limits[0]][sample_size].keys())
    data = []

    for class_label in classes:
        row = []
        for limit in limits:
            coverage = coverage_results[limit][sample_size].get(class_label, [0])[0]
            accuracy = accuracy_results[limit][sample_size].get(class_label, [0])[0]
            missed = missed_results[limit][sample_size].get(class_label, [0])[0]
            row.append(f"Cov: {coverage:.2f}%\nAcc: {accuracy:.2f}%\nMiss: {missed:.2f}%")
        data.append(row)

    columns = [f"Percentile_{p}" for p in percentiles]
    index = [("Total" if c == -1 else f"Class_{c}") for c in classes]
    df = pd.DataFrame(data, columns=columns, index=index)
    df.to_csv(f"/home/ev357/tcbench/src/fingerprinting/results/{name}.csv")

coverage_results_baseline_limits_per_class_splits_avg, accuracy_results_baseline_limits_per_class_splits_avg, missed_results_baseline_limits_per_class_splits_avg = extract_results_limits_per_class_splits_avg('baseline', 'euclidean', 'distance', samples, percentiles)
plot_results_per_class(coverage_results_baseline_limits_per_class_splits_avg, accuracy_results_baseline_limits_per_class_splits_avg, samples, percentiles, 'automated_per_class_baseline_splits_avg', percentile=True)
save_per_class_results_to_csv(coverage_results_baseline_limits_per_class_splits_avg, accuracy_results_baseline_limits_per_class_splits_avg, missed_results_baseline_limits_per_class_splits_avg, percentiles, 10, "automated_per_class_baseline_10_splits_avg")
save_per_class_results_to_csv(coverage_results_baseline_limits_per_class_splits_avg, accuracy_results_baseline_limits_per_class_splits_avg, missed_results_baseline_limits_per_class_splits_avg, percentiles, 100, "automated_per_class_baseline_100_splits_avg")
save_per_class_results_to_csv(coverage_results_baseline_limits_per_class_splits_avg, accuracy_results_baseline_limits_per_class_splits_avg, missed_results_baseline_limits_per_class_splits_avg, percentiles, 1000, "automated_per_class_baseline_1000_splits_avg")

coverage_results_embeddings_limits_per_class_splits_avg, accuracy_results_embeddings_limits_per_class_splits_avg, missed_results_embeddings_limits_per_class_splits_avg = extract_results_limits_per_class_splits_avg('embeddings', 'euclidean', 'distance', samples, percentiles)
plot_results_per_class(coverage_results_embeddings_limits_per_class_splits_avg, accuracy_results_embeddings_limits_per_class_splits_avg, samples, percentiles, 'automated_per_class_embeddings_splits_avg', percentile=True)
save_per_class_results_to_csv(coverage_results_embeddings_limits_per_class_splits_avg, accuracy_results_embeddings_limits_per_class_splits_avg, missed_results_embeddings_limits_per_class_splits_avg, percentiles, 10, "automated_per_class_embeddings_10_splits_avg")
save_per_class_results_to_csv(coverage_results_embeddings_limits_per_class_splits_avg, accuracy_results_embeddings_limits_per_class_splits_avg, missed_results_embeddings_limits_per_class_splits_avg, percentiles, 100, "automated_per_class_embeddings_100_splits_avg")
save_per_class_results_to_csv(coverage_results_embeddings_limits_per_class_splits_avg, accuracy_results_embeddings_limits_per_class_splits_avg, missed_results_embeddings_limits_per_class_splits_avg, percentiles, 1000, "automated_per_class_embeddings_1000_splits_avg")

coverage_results_embeddings_cosine_limits_per_class_splits_avg, accuracy_results_embeddings_cosine_limits_per_class_splits_avg, missed_results_embeddings_cosine_limits_per_class_splits_avg = extract_results_limits_per_class_splits_avg('embeddings', 'cosine', 'similarity', samples, percentiles)
plot_results_per_class(coverage_results_embeddings_cosine_limits_per_class_splits_avg, accuracy_results_embeddings_cosine_limits_per_class_splits_avg, samples, percentiles, 'automated_per_class_embeddings_cosine_splits_avg', percentile=True)
save_per_class_results_to_csv(coverage_results_embeddings_cosine_limits_per_class_splits_avg, accuracy_results_embeddings_cosine_limits_per_class_splits_avg, missed_results_embeddings_cosine_limits_per_class_splits_avg, percentiles, 10, "automated_per_class_embeddings_cosine_10_splits_avg")
save_per_class_results_to_csv(coverage_results_embeddings_cosine_limits_per_class_splits_avg, accuracy_results_embeddings_cosine_limits_per_class_splits_avg, missed_results_embeddings_cosine_limits_per_class_splits_avg, percentiles, 100, "automated_per_class_embeddings_cosine_100_splits_avg")
save_per_class_results_to_csv(coverage_results_embeddings_cosine_limits_per_class_splits_avg, accuracy_results_embeddings_cosine_limits_per_class_splits_avg, missed_results_embeddings_cosine_limits_per_class_splits_avg, percentiles, 1000, "automated_per_class_embeddings_cosine_1000_splits_avg")
