#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib.pyplot as plt
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


def plot_distance_histograms(class_distances_1, class_distances_2, name):
    fig, axes = plt.subplots(4, 5, figsize=(20, 15))
    axes = axes.flatten()

    for i, ((label1, distances1), (label2, distances2)) in enumerate(zip(class_distances_1.items(), class_distances_2.items())):
        ax = axes[i]
        distances1 = distances1[np.triu_indices_from(distances1, k=1)]
        distances2 = distances2.flatten()
        
        ax.hist(distances1, bins=30, alpha=0.5, color='blue', label='Within class', density=True)
        ax.hist(distances2, bins=30, alpha=0.5, color='red', label='Between classes', density=True)
        
        ax.set_title(f'Class {label1}')
        ax.set_xlabel('Distance')
        ax.set_ylabel('Density')
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"/home/ev357/tcbench/src/fingerprinting/plots/{name}.png")

within_class_file = "/home/ev357/rds/hpc-work/baseline_within.pkl"
between_class_file = "/home/ev357/rds/hpc-work/baseline_between.pkl"

with open(within_class_file, 'rb') as f:
    distances_within_baseline = pickle.load(f)
with open(between_class_file, 'rb') as f:
    distances_between_baseline = pickle.load(f)

within_class_file = "/home/ev357/rds/hpc-work/embeddings_within.npy"
between_class_file = "/home/ev357/rds/hpc-work/embeddings_between.npy"

with open(within_class_file, 'rb') as f:
    distances_within_embeddings = pickle.load(f)
with open(between_class_file, 'rb') as f:
    distances_between_embeddings = pickle.load(f)

within_class_file = "/home/ev357/rds/hpc-work/embeddings_cosine_within.npy"
between_class_file = "/home/ev357/rds/hpc-work/embeddings_cosine_between.npy"

with open(within_class_file, 'rb') as f:
    distances_within_embeddings_cosine = pickle.load(f)
with open(between_class_file, 'rb') as f:
    distances_between_embeddings_cosine = pickle.load(f)

within_class_file = "/home/ev357/rds/hpc-work/embeddings_cosine_similarity_within.npy"
between_class_file = "/home/ev357/rds/hpc-work/embeddings_cosine_similarity_between.npy"

with open(within_class_file, 'rb') as f:
    similarity_within_embeddings_cosine = pickle.load(f)
with open(between_class_file, 'rb') as f:
    similarity_between_embeddings_cosine = pickle.load(f)

plot_distance_histograms(distances_within_baseline, distances_between_baseline, "distances_baseline")
plot_distance_histograms(distances_within_embeddings, distances_between_embeddings, "distances_embeddings")
plot_distance_histograms(distances_within_embeddings_cosine, distances_between_embeddings_cosine, "distances_embeddings_cosine")
plot_distance_histograms(similarity_within_embeddings_cosine, similarity_between_embeddings_cosine, "distances_embeddings_cosine_similarity")


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

def plot_min_histograms(min_within, min_between, name, xlim=1):
    fig, axes = plt.subplots(4, 5, figsize=(20, 15))
    axes = axes.flatten()

    for i, (label, min_within_distances) in enumerate(min_within.items()):
        ax = axes[i]
        min_between_distances = min_between[label]
        
        ax.hist(min_within_distances, bins=30, alpha=0.5, color='blue', label='Min within', density=True)
        ax.hist(min_between_distances, bins=30, alpha=0.5, color='red', label='Min between', density=True)
        
        ax.set_title(f'Class {label}')
        ax.set_xlabel('Distance')
        ax.set_ylabel('Density')
        ax.set_xlim(0, xlim)
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"/home/ev357/tcbench/src/fingerprinting/plots/{name}.png")


def plot_max_histograms(max_within, max_between, name, xlim=0):
    fig, axes = plt.subplots(4, 5, figsize=(20, 15))
    axes = axes.flatten()

    for i, (label, max_within_distances) in enumerate(max_within.items()):
        ax = axes[i]
        max_between_distances = max_between[label]
        ax.hist(max_within_distances, bins=30, alpha=0.5, color='blue', label='Max within', density=True)
        ax.hist(max_between_distances, bins=30, alpha=0.5, color='red', label='Max between', density=True)
        
        ax.set_title(f'Class {label}')
        ax.set_xlabel('Distance')
        ax.set_ylabel('Density')
        ax.set_xlim(xlim, 1)
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"/home/ev357/tcbench/src/fingerprinting/plots/{name}.png")

min_within_baseline = find_min_within(distances_within_baseline)
min_between_baseline = find_min_between(distances_between_baseline)
min_within_embeddings = find_min_within(distances_within_embeddings)
min_between_embeddings = find_min_between(distances_between_embeddings)
min_within_embeddings_cosine = find_min_within(distances_within_embeddings_cosine)
min_between_embeddings_cosine = find_min_between(distances_between_embeddings_cosine)
max_within_embeddings_cosine_similarity = find_max_within(similarity_within_embeddings_cosine)
max_between_embeddings_cosine_similarity = find_max_between(similarity_between_embeddings_cosine)

plot_min_histograms(min_within_baseline, min_between_baseline, name="min_baseline")
plot_min_histograms(min_within_embeddings, min_between_embeddings, "min_embeddings")
plot_min_histograms(min_within_embeddings_cosine, min_between_embeddings_cosine, name="min_embeddings_cosine", xlim=0.1)
plot_max_histograms(max_within_embeddings_cosine_similarity, max_between_embeddings_cosine_similarity, name="max_embeddings_cosine_similarity", xlim=0.9)