#!/usr/bin/env python3
import pandas as pd
from functions import (extract_results_per_class,
                       extract_results_per_class_splits_avg,
                       plot_results_per_class)

samples = [10, 100, 1000]

distances_baseline = [0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3]
distances_embeddings = [0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
similarities = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 
                0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999]

coverage_results_baseline_per_class, accuracy_results_baseline_per_class = extract_results_per_class('baseline', 'euclidean', 'distance', samples, distances_baseline)
plot_results_per_class(coverage_results_baseline_per_class, accuracy_results_baseline_per_class, samples, distances_baseline, 'manual_per_class_baseline')

coverage_results_embeddings_per_class, accuracy_results_embeddings_per_class = extract_results_per_class('embeddings', 'euclidean', 'distance', samples, distances_embeddings)
plot_results_per_class(coverage_results_embeddings_per_class, accuracy_results_embeddings_per_class, samples, distances_embeddings, 'manual_per_class_embeddings')

coverage_results_embeddings_cosine_per_class, accuracy_results_embeddings_cosine_per_class = extract_results_per_class('embeddings', 'cosine', 'similarity', samples, similarities)
plot_results_per_class(coverage_results_embeddings_cosine_per_class, accuracy_results_embeddings_cosine_per_class, samples, similarities, 'manual_per_class_embeddings_cosine', reverse=True)


coverage_results_baseline_per_class_splits_avg, accuracy_results_baseline_per_class_splits_avg = extract_results_per_class_splits_avg('baseline', 'euclidean', 'distance', samples, distances_baseline)
plot_results_per_class(coverage_results_baseline_per_class_splits_avg, accuracy_results_baseline_per_class_splits_avg, samples, distances_baseline, 'manual_per_class_baseline_splits_avg')

coverage_results_embeddings_per_class_splits_avg, accuracy_results_embeddings_per_class_splits_avg = extract_results_per_class_splits_avg('embeddings', 'euclidean', 'distance', samples, distances_embeddings)
plot_results_per_class(coverage_results_embeddings_per_class_splits_avg, accuracy_results_embeddings_per_class_splits_avg, samples, distances_embeddings, 'manual_per_class_embeddings_splits_avg')

coverage_results_embeddings_cosine_per_class_splits_avg, accuracy_results_embeddings_cosine_per_class_splits_avg = extract_results_per_class_splits_avg('embeddings', 'cosine', 'similarity', samples, similarities)
plot_results_per_class(coverage_results_embeddings_cosine_per_class_splits_avg, accuracy_results_embeddings_cosine_per_class_splits_avg, samples, similarities, 'manual_per_class_embeddings_cosine_splits_avg', reverse=True)


def save_per_class_results_to_csv(coverage_results, accuracy_results, limits, metric, sample_size, name):
    classes = sorted(coverage_results[limits[0]][sample_size].keys())
    data = []

    for class_label in classes:
        row = []
        for limit in limits:
            coverage = coverage_results[limit][sample_size].get(class_label, [0])[0]
            accuracy = accuracy_results[limit][sample_size].get(class_label, [0])[0]
            row.append(f"Cov: {coverage:.2f}%\nAcc: {accuracy:.2f}%")
        data.append(row)

    if metric == "distance":
        columns = [f"Distance_{d}" for d in limits]
    elif metric == "similarity":
        columns = [f"Similarity_{s}" for s in limits]
    index = [("Total" if c == -1 else f"Class_{c}") for c in classes]
    df = pd.DataFrame(data, columns=columns, index=index)
    df.to_csv(f"/home/ev357/tcbench/src/fingerprinting/results/{name}.csv")


save_per_class_results_to_csv(coverage_results_baseline_per_class_splits_avg, accuracy_results_baseline_per_class_splits_avg, distances_baseline, "distance", 10, "manual_per_class_baseline_10_splits_avg")
save_per_class_results_to_csv(coverage_results_baseline_per_class_splits_avg, accuracy_results_baseline_per_class_splits_avg, distances_baseline, "distance", 100, "manual_per_class_baseline_100_splits_avg")
save_per_class_results_to_csv(coverage_results_baseline_per_class_splits_avg, accuracy_results_baseline_per_class_splits_avg, distances_baseline, "distance", 1000, "manual_per_class_baseline_1000_splits_avg")

save_per_class_results_to_csv(coverage_results_embeddings_per_class_splits_avg, accuracy_results_embeddings_per_class_splits_avg, distances_embeddings, "distance", 10, "manual_per_class_embeddings_10_splits_avg")
save_per_class_results_to_csv(coverage_results_embeddings_per_class_splits_avg, accuracy_results_embeddings_per_class_splits_avg, distances_embeddings, "distance", 100, "manual_per_class_embeddings_100_splits_avg")
save_per_class_results_to_csv(coverage_results_embeddings_per_class_splits_avg, accuracy_results_embeddings_per_class_splits_avg, distances_embeddings, "distance", 1000, "manual_per_class_embeddings_1000_splits_avg")

save_per_class_results_to_csv(coverage_results_embeddings_cosine_per_class_splits_avg, accuracy_results_embeddings_cosine_per_class_splits_avg, similarities, "similarity", 10, "manual_per_class_embeddings_cosine_10_splits_avg")
save_per_class_results_to_csv(coverage_results_embeddings_cosine_per_class_splits_avg, accuracy_results_embeddings_cosine_per_class_splits_avg, similarities, "similarity", 100, "manual_per_class_embeddings_cosine_100_splits_avg")
save_per_class_results_to_csv(coverage_results_embeddings_cosine_per_class_splits_avg, accuracy_results_embeddings_cosine_per_class_splits_avg, similarities, "similarity", 1000, "manual_per_class_embeddings_cosine_1000_splits_avg")