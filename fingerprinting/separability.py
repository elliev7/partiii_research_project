#!/usr/bin/env python3
import numpy as np
import json
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples

vectors_baseline = np.load('/home/ev357/tcbench/src/fingerprinting/mirage19/baseline_vectors.npy')
labels_baseline = np.load('/home/ev357/tcbench/src/fingerprinting/mirage19/baseline_labels.npy')
vectors_embeddings = np.load('/home/ev357/tcbench/src/fingerprinting/mirage19/embeddings_vectors.npy')
labels_embeddings = np.load('/home/ev357/tcbench/src/fingerprinting/mirage19/embeddings_labels.npy')

def compute_cluster_metrics(X, labels, metric='euclidean'):
    silhouette_global = float(silhouette_score(X, labels, metric=metric))
    silhouette_vals = silhouette_samples(X, labels, metric=metric)
    dbi = float(davies_bouldin_score(X, labels))

    unique_labels = np.unique(labels)
    per_class_silhouette = {
        str(label): float(np.mean(silhouette_vals[labels == label]))
        for label in unique_labels
    }

    return {
        'silhouette': silhouette_global,
        'dbi': dbi,
        'per_class_silhouette': per_class_silhouette
    }

baseline_scores = compute_cluster_metrics(vectors_baseline, labels_baseline)
embedding_scores = compute_cluster_metrics(vectors_embeddings, labels_embeddings)

output = {
    'baseline': baseline_scores,
    'embeddings': embedding_scores
}

output_path = '/home/ev357/tcbench/src/fingerprinting/separability/scores.json'
with open(output_path, 'w') as f:
    json.dump(output, f, indent=4)
