#!/usr/bin/env python3
import pickle
import os
from functions import extract_results_limits_per_class_iterative_splits_avg

samples = [10, 100, 1000]
percentiles = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]

output_dir = "/home/ev357/rds/hpc-work/results/iterative/"
os.makedirs(output_dir, exist_ok=True)

cov_euc, acc_euc, preds_euc, trues_euc = extract_results_limits_per_class_iterative_splits_avg(
    'embeddings', 'euclidean', 'distance', samples, percentiles
)
with open(f"{output_dir}/embeddings_euclidean.pkl", "wb") as f:
    pickle.dump((cov_euc, acc_euc, preds_euc, trues_euc), f)

cov_cos, acc_cos, preds_cos, trues_cos = extract_results_limits_per_class_iterative_splits_avg(
    'embeddings', 'cosine', 'similarity', samples, percentiles
)
with open(f"{output_dir}/embeddings_cosine.pkl", "wb") as f:
    pickle.dump((cov_cos, acc_cos, preds_cos, trues_cos), f)