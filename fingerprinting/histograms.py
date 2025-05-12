#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib.pyplot as plt

vectors_baseline = np.load('/home/ev357/tcbench/src/fingerprinting/artifacts-mirage19/baseline_vectors.npy')
labels_baseline = np.load('/home/ev357/tcbench/src/fingerprinting/artifacts-mirage19/baseline_labels.npy')
vectors_embeddings = np.load('/home/ev357/tcbench/src/fingerprinting/artifacts-mirage19/embeddings_vectors.npy')
labels_embeddings = np.load('/home/ev357/tcbench/src/fingerprinting/artifacts-mirage19/embeddings_labels.npy')

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