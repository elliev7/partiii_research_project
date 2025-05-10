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



def find_min_within(within_class_distances):
    min_distances = {}
    for label, distances in within_class_distances.items():
        np.fill_diagonal(distances, np.inf)
        min_distances[label] = np.min(distances, axis=1)
    return min_distances

def find_min_between(between_class_distances):
    min_distances = {}
    for label, distances in between_class_distances.items():
        min_distances[label] = np.min(distances, axis=1)
    return min_distances

def find_max_within(within_class_distances):
    max_distances = {}
    for label, distances in within_class_distances.items():
        distances = np.where(np.isinf(distances), np.nan, distances)
        np.fill_diagonal(distances, np.nan)
        max_distances[label] = np.nanmax(distances, axis=1)
    return max_distances

def find_max_between(between_class_distances):
    max_distances = {}
    for label, distances in between_class_distances.items():
        distances = np.where(np.isinf(distances), np.nan, distances)
        max_distances[label] = np.nanmax(distances, axis=1)
    return max_distances


within_class_file = "/home/ev357/rds/hpc-work/baseline_min_within.pkl"
between_class_file = "/home/ev357/rds/hpc-work/baseline_min_between.pkl"

min_within_baseline = find_min_within(distances_within_baseline)
min_between_baseline = find_min_between(distances_between_baseline)

with open(within_class_file, 'wb') as f:
    pickle.dump(min_within_baseline, f)
with open(between_class_file, 'wb') as f:
    pickle.dump(min_between_baseline, f)


within_class_file = "/home/ev357/rds/hpc-work/embeddings_min_within.npy"
between_class_file = "/home/ev357/rds/hpc-work/embeddings_min_between.npy"

min_within_embeddings = find_min_within(distances_within_embeddings)
min_between_embeddings = find_min_between(distances_between_embeddings)

with open(within_class_file, 'wb') as f:
    pickle.dump(min_within_embeddings, f)
with open(between_class_file, 'wb') as f:
    pickle.dump(min_between_embeddings, f)


within_class_file = "/home/ev357/rds/hpc-work/embeddings_cosine_min_within.npy"
between_class_file = "/home/ev357/rds/hpc-work/embeddings_cosine_min_between.npy"

min_within_embeddings_cosine = find_min_within(distances_within_embeddings_cosine)
min_between_embeddings_cosine = find_min_between(distances_between_embeddings_cosine)

with open(within_class_file, 'wb') as f:
    pickle.dump(min_within_embeddings_cosine, f)
with open(between_class_file, 'wb') as f:
    pickle.dump(min_between_embeddings_cosine, f)


within_class_file = "/home/ev357/rds/hpc-work/embeddings_cosine_similarity_max_within.npy"
between_class_file = "/home/ev357/rds/hpc-work/embeddings_cosine_similarity_max_between.npy"

max_within_embeddings_cosine_similarity = find_max_within(similarity_within_embeddings_cosine)
max_between_embeddings_cosine_similarity = find_max_between(similarity_between_embeddings_cosine)

with open(within_class_file, 'wb') as f:
    pickle.dump(max_within_embeddings_cosine_similarity, f)
with open(between_class_file, 'wb') as f:
    pickle.dump(max_between_embeddings_cosine_similarity, f)