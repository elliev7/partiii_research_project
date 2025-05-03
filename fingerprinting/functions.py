import numpy as np
import matplotlib.pyplot as plt
import faiss


def build_faiss_index(data, labels, train_indices, distance_type, samples_per_class = None, seed=42):
    np.random.seed(seed)

    filtered_data = data[train_indices]
    filtered_labels = labels[train_indices]
    d = filtered_data.shape[1]

    selected_data = []
    selected_indices = []
    unique_labels = np.unique(filtered_labels)
    for label in unique_labels:
        label_indices = np.where(filtered_labels == label)[0]
        if samples_per_class is None or samples_per_class == -1:
            selected_label_indices = label_indices
        else:
            selected_label_indices = np.random.choice(label_indices, size=min(samples_per_class, len(label_indices)), replace=False)
        selected_data.append(filtered_data[selected_label_indices])
        selected_indices.extend(train_indices[selected_label_indices])
    selected_data = np.vstack(selected_data)
    selected_indices = np.array(selected_indices)

    if distance_type == 'euclidean':
        index = faiss.IndexFlatL2(d)
    elif distance_type == 'cosine':
        selected_data = selected_data / np.linalg.norm(selected_data, axis=1, keepdims=True)
        index = faiss.IndexFlatIP(d)
    else:
        raise ValueError("Unsupported distance type. Use 'euclidean' or 'cosine'.")

    index.add(selected_data)
    return index, selected_indices

def search_and_compare_labels(data, labels, test_indices, selected_indices, index, metric, limit=None):
    k = 1
    if metric == "distance":
        D, I = index.search(data[test_indices], k)
    elif metric == "similarity":
        query_vectors = data[test_indices]
        query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
        D, I = index.search(query_vectors, k)
    
    test_labels = labels[test_indices]
    neighbor_labels = labels[selected_indices[I.flatten()]].reshape(I.shape)
    
    matches = 0
    classified = 0
    min_distances = []

    for i in range(len(test_labels)):  
        if metric == "distance":
            min_distances.append(np.sqrt(D[i,0]))
            if limit is not None and np.sqrt(D[i, 0]) > limit:
                continue
        if metric == "similarity":
            min_distances.append(D[i,0])
            if limit is not None and D[i, 0] < limit:
                continue
        classified += 1
        matches += test_labels[i] in neighbor_labels[i, :1]

    classified_percentage = (classified / len(test_labels)) * 100
    match_percentage = (matches / classified) * 100 if classified > 0 else 0
    return classified_percentage, match_percentage, min_distances

def search_and_compare_labels_per_class(data, labels, test_indices, selected_indices, index, metric, limit=None):
    k = 1
    if metric == "distance":
        D, I = index.search(data[test_indices], k)
    elif metric == "similarity":
        query_vectors = data[test_indices]
        query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
        D, I = index.search(query_vectors, k)
    
    test_labels = labels[test_indices]
    neighbor_labels = labels[selected_indices[I.flatten()]].reshape(I.shape)
    
    unique_labels = np.unique(test_labels)
    per_class_results = {label: {"classified": 0, "matches": 0, "total": 0} for label in unique_labels}

    for i in range(len(test_labels)):
        label = test_labels[i]
        per_class_results[label]["total"] += 1

        if metric == "distance" and limit is not None and np.sqrt(D[i, 0]) > limit:
            continue
        if metric == "similarity" and limit is not None and D[i, 0] < limit:
            continue

        per_class_results[label]["classified"] += 1
        per_class_results[label]["matches"] += test_labels[i] in neighbor_labels[i, :1]

    per_class_metrics = {}
    for label, results in per_class_results.items():
        classified = results["classified"]
        total = results["total"]
        matches = results["matches"]

        classified_percentage = (classified / total) * 100 if total > 0 else 0
        match_percentage = (matches / classified) * 100 if classified > 0 else 0
        per_class_metrics[label] = {
            "classified_percentage": classified_percentage,
            "match_percentage": match_percentage,
        }

    return per_class_metrics

def extract_results(vectors, labels, train_indices, test_indices, distance_type, metric, samples, limits):
    classified_results = {li: [] for li in limits}
    accuracy_results = {li: [] for li in limits}

    for limit in limits:
        for sample_size in samples:
            index, selected_indices = build_faiss_index(
                vectors, labels, train_indices, distance_type, sample_size
            )
            classified, accuracy, min_distances = search_and_compare_labels(
                vectors, labels, test_indices, selected_indices, index, metric, limit
            )
            classified_results[limit].append(classified)
            accuracy_results[limit].append(accuracy)   
    return classified_results, accuracy_results, min_distances

def extract_results_splits(vectors, labels, train_test_splits, distance_type, metric, samples, limits):
    accuracy_results = {sample_size: {limit: [] for limit in limits} for sample_size in samples}
    classified_results = {sample_size: {limit: [] for limit in limits} for sample_size in samples}

    for train_indices, test_indices in train_test_splits:
        for seed in range(40,50):
            for sample_size in samples:
                for limit in limits:
                    index, selected_indices = build_faiss_index(
                        vectors, labels, train_indices, distance_type, sample_size, seed
                    )
                    classified, accuracy, _ = search_and_compare_labels(
                        vectors, labels, test_indices, selected_indices, index, metric, limit
                    )
                    accuracy_results[sample_size][limit].append(accuracy)
                    classified_results[sample_size][limit].append(classified) 
    return classified_results, accuracy_results

def extract_results_per_class(vectors, labels, train_indices, test_indices, distance_type, metric, samples, limits):
    per_class_classified_results = {li: {sample: {} for sample in samples} for li in limits}
    per_class_accuracy_results = {li: {sample: {} for sample in samples} for li in limits}

    for limit in limits:
        for sample_size in samples:
            index, selected_indices = build_faiss_index(
                vectors, labels, train_indices, distance_type, sample_size
            )
            per_class_metrics = search_and_compare_labels_per_class(
                vectors, labels, test_indices, selected_indices, index, metric, limit
            )

            for label, metrics in per_class_metrics.items():
                if label not in per_class_classified_results[limit][sample_size]:
                    per_class_classified_results[limit][sample_size][label] = []
                    per_class_accuracy_results[limit][sample_size][label] = []

                per_class_classified_results[limit][sample_size][label].append(metrics["classified_percentage"])
                per_class_accuracy_results[limit][sample_size][label].append(metrics["match_percentage"])

    return per_class_classified_results, per_class_accuracy_results

def plot_results_by_distance(classified_results, accuracy_results, samples, distances):
    num_samples = len(samples)
    fig, axes = plt.subplots(num_samples, 1, figsize=(6, 4 * num_samples), sharex=True)
    axes = axes.flatten() if num_samples > 1 else [axes]

    for i, distance in enumerate(distances):
        ax = axes[i]
        
        ax.plot(samples, accuracy_results[distance], label='Accuracy', color='blue')
        ax.set_xlabel('Number of Samples Per Class') 
        ax.set_ylabel('Accuracy (%)', color='blue')   
        ax.tick_params(axis='y', labelcolor='blue')
        ax.set_ylim(0, 100)
        ax.set_title(f'Distance: {distance}')
        ax.grid(True)
        
        ax2 = ax.twinx()
        ax2.plot(samples, classified_results[distance], label='Classified %', color='green', linestyle='--')
        ax2.set_xlabel('Number of Samples Per Class')
        ax2.set_ylabel('Coverage (%)', color='green')
        ax2.set_ylim(0, 100)
        ax2.tick_params(axis='y', labelcolor='green')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.show()

def plot_results_by_sample_size(classified_results, accuracy_results, samples, distances):
    num_samples = len(samples)
    fig, axes = plt.subplots(num_samples, 1, figsize=(6, 4 * num_samples), sharex=True)
    axes = axes.flatten() if num_samples > 1 else [axes]

    for i, sample_size in enumerate(samples):
        ax = axes[i]

        classified = [classified_results[distance][i] for distance in distances]
        accuracy = [accuracy_results[distance][i] for distance in distances]

        ax.plot(distances, accuracy, label='Accuracy', color='blue')
        ax.set_xlabel('Distance')
        ax.set_ylabel('Accuracy (%)', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax.set_ylim(0, 100)
        ax.set_title(f'Sample Size: {sample_size}')
        ax.grid(True)

        ax2 = ax.twinx()
        ax2.plot(distances, classified, label='Classified %', color='green', linestyle='--')
        ax2.set_ylabel('Coverage (%)', color='green')
        ax2.set_ylim(0, 100)
        ax2.tick_params(axis='y', labelcolor='green')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def plot_results_by_sample_size_splits(classified_results, accuracy_results, samples, distances):
    num_samples = len(samples)
    fig, axes = plt.subplots(num_samples, 1, figsize=(6, 4 * num_samples), sharex=True)
    axes = axes.flatten() if num_samples > 1 else [axes]

    for i, sample_size in enumerate(samples):
        ax = axes[i]
        ax2 = ax.twinx()

        for run_idx in range(len(accuracy_results[sample_size][distances[0]])):
            accuracy = [accuracy_results[sample_size][distance][run_idx] for distance in distances]
            classified = [classified_results[sample_size][distance][run_idx] for distance in distances]

            ax.plot(distances, accuracy, color='blue', alpha=0.3, linewidth=0.5, label='Accuracy' if run_idx == 0 else None)
            ax2.plot(distances, classified, color='green', alpha=0.3, linestyle='--', linewidth=0.5, label='Classified %' if run_idx == 0 else None)

        ax.set_xlabel('Distance')
        ax.set_ylabel('Accuracy (%)', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax.set_ylim(0, 100)
        ax.set_title(f'Sample Size: {sample_size}')
        ax.grid(True)

        ax2.set_ylabel('Coverage (%)', color='green')
        ax2.set_ylim(0, 100)
        ax2.tick_params(axis='y', labelcolor='green')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def plot_results_by_sample_size_per_class(per_class_classified_results, per_class_accuracy_results, samples, distances):
    num_samples = len(samples)
    fig, axes = plt.subplots(num_samples, 1, figsize=(6, 4 * num_samples), sharex=True)
    axes = axes.flatten() if num_samples > 1 else [axes]

    unique_classes = set()
    for limit_results in per_class_classified_results.values():
        for sample_results in limit_results.values():
            unique_classes.update(sample_results.keys())
    unique_classes = sorted(unique_classes)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))

    for i, sample_size in enumerate(samples):
        ax = axes[i]
        ax2 = ax.twinx()

        for class_idx, class_label in enumerate(unique_classes):
            classified = [
                per_class_classified_results[distance][sample_size].get(class_label, [0])[0]
                for distance in distances
            ]
            accuracy = [
                per_class_accuracy_results[distance][sample_size].get(class_label, [0])[0]
                for distance in distances
            ]

            ax.plot(distances, accuracy, label=f'Class {class_label} Accuracy', color=colors[class_idx])
            ax.set_xlabel('Distance')
            ax.set_ylabel('Accuracy (%) —')
            ax.tick_params(axis='y')
            ax.set_ylim(0, 100)
            ax.set_title(f'Sample Size: {sample_size}')
            ax.grid(True)

            ax2.plot(distances, classified, label=f'Class {class_label} Classified %', color=colors[class_idx], linestyle='--')
        
        ax2.set_ylabel('Coverage (%) --')
        ax2.set_ylim(0, 100)
        ax2.tick_params(axis='y')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()