#!/usr/bin/env python3
import pickle
import numpy as np

from scipy.spatial.distance import cdist, pdist, squareform

vectors_baseline = np.load('/home/ev357/tcbench/src/fingerprinting/artifacts-mirage19/baseline_vectors.npy')
labels_baseline = np.load('/home/ev357/tcbench/src/fingerprinting/artifacts-mirage19/baseline_labels.npy')
vectors_embeddings = np.load('/home/ev357/tcbench/src/fingerprinting/artifacts-mirage19/embeddings_vectors.npy')
labels_embeddings = np.load('/home/ev357/tcbench/src/fingerprinting/artifacts-mirage19/embeddings_labels.npy')

def calculate_within_class_distances(feature_matrix, true_labels, metric):
    class_distances = {}
    unique_labels = np.unique(true_labels)

    for label in unique_labels:
        class_indices = np.where(true_labels == label)[0]
        class_features = feature_matrix[class_indices]
        distances = squareform(pdist(class_features, metric))
        class_distances[label] = distances

    return class_distances

def calculate_between_class_distances(feature_matrix, true_labels, metric):
    class_distances = {}
    unique_labels = np.unique(true_labels)

    for label in unique_labels:
        class_indices = np.where(true_labels == label)[0]
        other_indices = np.where(true_labels != label)[0]
        class_features = feature_matrix[class_indices]
        other_features = feature_matrix[other_indices]
        distances = cdist(class_features, other_features, metric)
        class_distances[label] = distances

    return class_distances


within_class_file = "/home/ev357/rds/hpc-work/baseline_within.pkl"
between_class_file = "/home/ev357/rds/hpc-work/baseline_between.pkl"

distances_within_baseline = calculate_within_class_distances(vectors_baseline, labels_baseline, metric='euclidean')
distances_between_baseline = calculate_between_class_distances(vectors_baseline, labels_baseline, metric='euclidean')
    
with open(within_class_file, 'wb') as f:
    pickle.dump(distances_within_baseline, f)
with open(between_class_file, 'wb') as f:
    pickle.dump(distances_between_baseline, f)


within_class_file = "/home/ev357/rds/hpc-work/embeddings_within.npy"
between_class_file = "/home/ev357/rds/hpc-work/embeddings_between.npy"

distances_within_embeddings = calculate_within_class_distances(vectors_embeddings, labels_embeddings, metric='euclidean')
distances_between_embeddings = calculate_between_class_distances(vectors_embeddings, labels_embeddings, metric='euclidean')
    
with open(within_class_file, 'wb') as f:
    pickle.dump(distances_within_embeddings, f)
with open(between_class_file, 'wb') as f:
    pickle.dump(distances_between_embeddings, f)


within_class_file = "/home/ev357/rds/hpc-work/embeddings_cosine_within.npy"
between_class_file = "/home/ev357/rds/hpc-work/embeddings_cosine_between.npy"

distances_within_embeddings_cosine = calculate_within_class_distances(vectors_embeddings, labels_embeddings, metric='cosine')
distances_between_embeddings_cosine = calculate_between_class_distances(vectors_embeddings, labels_embeddings, metric='cosine')
    
with open(within_class_file, 'wb') as f:
    pickle.dump(distances_within_embeddings_cosine, f)
with open(between_class_file, 'wb') as f:
    pickle.dump(distances_between_embeddings_cosine, f)


within_class_file = "/home/ev357/rds/hpc-work/embeddings_cosine_similarity_within.npy"
between_class_file = "/home/ev357/rds/hpc-work/embeddings_cosine_similarity_between.npy"

similarity_within_embeddings_cosine = {
    label: 1 - distances for label, distances in distances_within_embeddings_cosine.items()
}
similarity_between_embeddings_cosine = {
    label: 1 - distances for label, distances in distances_between_embeddings_cosine.items()
}
    
with open(within_class_file, 'wb') as f:
    pickle.dump(similarity_within_embeddings_cosine, f)
with open(between_class_file, 'wb') as f:
    pickle.dump(similarity_between_embeddings_cosine, f)
