import numpy as np
import matplotlib.pyplot as plt
import faiss

def build_faiss_index(data, labels, train_indices, distance_type, samples_per_class=None, exclude_class=None, seed=42):
    selected_indices = select_indices_by_class(
        labels, train_indices, samples_per_class, exclude_class=exclude_class, seed=seed
    )

    selected_data = data[selected_indices]
    d = selected_data.shape[1]

    if distance_type == 'euclidean':
        index = faiss.IndexFlatL2(d)
    elif distance_type == 'cosine':
        selected_data = selected_data / np.linalg.norm(selected_data, axis=1, keepdims=True)
        index = faiss.IndexFlatIP(d)
    else:
        raise ValueError("Unsupported distance type. Use 'euclidean' or 'cosine'.")

    index.add(selected_data)
    return index, selected_indices

def select_indices_by_class(labels, train_indices, samples_per_class, exclude_class=None, seed=42):
    np.random.seed(seed)
    filtered_labels = labels[train_indices]
    unique_labels = np.unique(filtered_labels)
    selected_indices = []
    for label in unique_labels:
        if exclude_class is not None and label == exclude_class:
            continue
        label_indices = np.where(filtered_labels == label)[0]
        if samples_per_class is None or samples_per_class == -1:
            selected_label_indices = label_indices
        else:
            selected_label_indices = np.random.choice(
                label_indices, size=min(samples_per_class, len(label_indices)), replace=False
            )
        selected_indices.extend(train_indices[selected_label_indices])
    return np.array(selected_indices)

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
    coverage = 0
    min_distances = []

    for i in range(len(test_indices)):  
        if metric == "distance":
            min_distances.append(np.sqrt(D[i,0]))
            if limit is not None and np.sqrt(D[i, 0]) > limit:
                continue
        if metric == "similarity":
            min_distances.append(D[i,0])
            if limit is not None and D[i, 0] < limit:
                continue
        coverage += 1
        matches += test_labels[i] == neighbor_labels[i, 0]

    coverage_percentage = (coverage / len(test_indices)) * 100
    match_percentage = (matches / coverage) * 100 if coverage > 0 else 0
    return coverage_percentage, match_percentage, min_distances

def search_and_compare_labels_majority(data, labels, test_indices, selected_indices, index, metric, limit=None):
    k = 10
    if metric == "distance":
        D, I = index.search(data[test_indices], k)
    elif metric == "similarity":
        query_vectors = data[test_indices]
        query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
        D, I = index.search(query_vectors, k)
    
    test_labels = labels[test_indices]
    neighbor_labels = labels[selected_indices[I.flatten()]].reshape(I.shape)
    
    matches = 0
    coverage = 0

    for i in range(len(test_indices)):  
        if metric == "distance":
            distances = np.sqrt(D[i])
            valid_indices = distances <= limit if limit is not None else np.arange(len(distances))
        elif metric == "similarity":
            distances = D[i]
            valid_indices = distances >= limit if limit is not None else np.arange(len(distances))
        
        valid_neighbors = neighbor_labels[i][valid_indices]
        valid_distances = distances[valid_indices]

        if len(valid_neighbors) == 0:
            continue

        unique_labels, counts = np.unique(valid_neighbors, return_counts=True)
        max_count = np.max(counts)
        majority_labels = unique_labels[counts == max_count]

        if len(majority_labels) > 1:
            closest_label = majority_labels[np.argmin([valid_distances[valid_neighbors == label][0] for label in majority_labels])]
        else:
            closest_label = majority_labels[0]

        coverage += 1
        matches += (test_labels[i] == closest_label)

    coverage_percentage = (coverage / len(test_indices)) * 100
    match_percentage = (matches / coverage) * 100 if coverage > 0 else 0
    return coverage_percentage, match_percentage

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
    per_class_results = {label: {"coverage": 0, "matches": 0, "total": 0} for label in unique_labels}
    per_class_results[-1] = {"coverage": 0, "matches": 0, "total": 0}

    for i in range(len(test_indices)):
        test_label = test_labels[i]
        per_class_results[test_label]["total"] += 1
        per_class_results[-1]["total"] += 1

        if metric == "distance" and limit is not None and np.sqrt(D[i, 0]) > limit:
            continue
        if metric == "similarity" and limit is not None and D[i, 0] < limit:
            continue

        per_class_results[test_label]["coverage"] += 1
        per_class_results[-1]["coverage"] += 1
        per_class_results[test_label]["matches"] += test_label == neighbor_labels[i, 0]
        per_class_results[-1]["matches"] += test_label == neighbor_labels[i, 0]

    per_class_metrics = {}
    for label, results in per_class_results.items():
        coverage = results["coverage"]
        total = results["total"]
        matches = results["matches"]

        coverage_percentage = (coverage / total) * 100 if total > 0 else 0
        match_percentage = (matches / coverage) * 100 if coverage > 0 else 0
        per_class_metrics[label] = {
            "coverage_percentage": coverage_percentage,
            "match_percentage": match_percentage,
        }

    return per_class_metrics

def search_and_compare_labels_limits_per_class(data, labels, test_indices, selected_indices, index, metric, class_limits):
    if metric == "distance":
        max_limit = max(class_limits.values())
        lims, D, I = index.range_search(data[test_indices], max_limit**2)

        sorted_D = np.zeros_like(D)
        sorted_I = np.zeros_like(I)

        for i in range(len(lims) - 1):
            start, end = lims[i], lims[i + 1]
            distances = D[start:end]
            indices = I[start:end]
            sorted_order = np.argsort(distances)
            
            sorted_D[start:end] = distances[sorted_order]
            sorted_I[start:end] = indices[sorted_order]

    elif metric == "similarity":
        min_limit = min(class_limits.values())
        query_vectors = data[test_indices]
        query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
        lims, D, I = index.range_search(query_vectors, min_limit)

        sorted_D = np.zeros_like(D)
        sorted_I = np.zeros_like(I)
        sorted_labels = []

        for i in range(len(lims) - 1):
            start, end = lims[i], lims[i + 1]
            distances = D[start:end]
            indices = I[start:end]
            
            sorted_order = np.argsort(distances)[::-1]
            
            sorted_D[start:end] = distances[sorted_order]
            sorted_I[start:end] = indices[sorted_order]
            
            sorted_labels.append(labels[sorted_I[start:end]])
    
    test_labels = labels[test_indices]    
    unique_labels = np.unique(test_labels)
    per_class_results = {label: {"coverage": 0, "matches": 0, "total": 0, "missed": 0} for label in unique_labels}
    per_class_results[-1] = {"coverage": 0, "matches": 0, "total": 0, "missed": 0}

    for i in range(len(test_labels)):
        test_class = test_labels[i]
        per_class_results[test_class]["total"] += 1
        per_class_results[-1]["total"] += 1

        start, end = lims[i], lims[i + 1]
        neighbors_distances = sorted_D[start:end]
        neighbors_indices = sorted_I[start:end]
        neighbors_classes = labels[selected_indices[sorted_I[start:end]]]
        
        if len(neighbors_distances) == 0:
            continue

        nn_distance = neighbors_distances[0]
        nn_index = neighbors_indices[0]
        nn_class = labels[selected_indices[nn_index]]
        nn_limit = class_limits.get(nn_class, None)
        test_class_limit = class_limits.get(test_class, None)

        if metric == "distance":
            if np.sqrt(nn_distance) <= nn_limit:
                per_class_results[test_class]["coverage"] += 1
                per_class_results[-1]["coverage"] += 1
                if nn_class == test_class:
                    per_class_results[test_class]["matches"] += 1
                    per_class_results[-1]["matches"] += 1
            else:
                for dist, n_class in zip(neighbors_distances[1:], neighbors_classes[1:]):
                    if n_class == test_class and np.sqrt(dist) <= test_class_limit:
                        per_class_results[test_class]["missed"] += 1
                        per_class_results[-1]["missed"] += 1
                        break
               
        elif metric == "similarity":
            if nn_distance >= nn_limit:
                per_class_results[test_class]["coverage"] += 1
                per_class_results[-1]["coverage"] += 1
                if nn_class == test_class:
                    per_class_results[test_class]["matches"] += 1
                    per_class_results[-1]["matches"] += 1
            else:
                for sim, n_class in zip(neighbors_distances[1:], neighbors_classes[1:]):
                    if n_class == test_class and sim >= test_class_limit:
                        per_class_results[test_class]["missed"] += 1
                        per_class_results[-1]["missed"] += 1
                        break

    per_class_metrics = {}
    for label, results in per_class_results.items():
        coverage = results["coverage"]
        missed = results["missed"]
        total = results["total"]
        matches = results["matches"]

        coverage_percentage = (coverage / total) * 100 if total > 0 else 0
        missed_percentage = (missed / total) * 100 if total > 0 else 0
        match_percentage = (matches / coverage) * 100 if coverage > 0 else 0
        per_class_metrics[label] = {
            "coverage_percentage": coverage_percentage,
            "missed_percentage": missed_percentage,
            "match_percentage": match_percentage,
        }
    
    return per_class_metrics

def calculate_limits_per_class(data, labels, train_indices, distance_type, samples_per_class, percentiles, seed=42):
    selected_indices = select_indices_by_class(
        labels, train_indices, samples_per_class, exclude_class=None, seed=seed 
    )
    selected_labels = labels[selected_indices]
    unique_labels = np.unique(selected_labels)
    class_nn_distances = {}

    for label in unique_labels:
        class_mask = selected_labels == label
        class_indices = selected_indices[class_mask]
        class_data = data[class_indices]
        d = class_data.shape[1]

        if distance_type == 'euclidean':
            index = faiss.IndexFlatL2(d)
        elif distance_type == 'cosine':
            class_data = class_data / np.linalg.norm(class_data, axis=1, keepdims=True)
            index = faiss.IndexFlatIP(d)
        index.add(class_data)

        k=2
        D, I = index.search(class_data, k)
        if distance_type == 'euclidean':
            nn_distances = np.sqrt(D[:, 1])
        elif distance_type == 'cosine':
            nn_distances = D[:, 1]

        class_nn_distances[label] = nn_distances

    limits_per_percentile = {}
    for percentile in percentiles:
        limits = {}
        for label, distances in class_nn_distances.items():
            if len(distances) == 0:
                limits[label] = 0
            else:
                limits[label] = np.percentile(distances, percentile)
        limits_per_percentile[percentile] = limits

    return limits_per_percentile

def extract_results(vectors, labels, train_indices, test_indices, distance_type, metric, samples, limits, exclude_class=None):
    coverage_results = {li: [] for li in limits}
    accuracy_results = {li: [] for li in limits}

    for limit in limits:
        for sample_size in samples:
            index, selected_indices = build_faiss_index(
                vectors, labels, train_indices, distance_type, sample_size, exclude_class=exclude_class
            )
            coverage, accuracy, min_distances = search_and_compare_labels(
                vectors, labels, test_indices, selected_indices, index, metric, limit
            )
            coverage_results[limit].append(coverage)
            accuracy_results[limit].append(accuracy)   
    return coverage_results, accuracy_results, min_distances

def extract_results_majority(vectors, labels, train_indices, test_indices, distance_type, metric, samples, limits):
    coverage_results = {li: [] for li in limits}
    accuracy_results = {li: [] for li in limits}

    for limit in limits:
        for sample_size in samples:
            index, selected_indices = build_faiss_index(
                vectors, labels, train_indices, distance_type, sample_size
            )
            coverage, accuracy = search_and_compare_labels_majority(
                vectors, labels, test_indices, selected_indices, index, metric, limit
            )
            coverage_results[limit].append(coverage)
            accuracy_results[limit].append(accuracy)   
    return coverage_results, accuracy_results

def extract_results_splits(vectors, labels, train_test_splits, distance_type, metric, samples, limits):
    accuracy_results = {sample_size: {limit: [] for limit in limits} for sample_size in samples}
    coverage_results = {sample_size: {limit: [] for limit in limits} for sample_size in samples}

    for train_indices, test_indices in train_test_splits:
        for seed in range(40, 50):
            for sample_size in samples:
                for limit in limits:
                    index, selected_indices = build_faiss_index(
                        vectors, labels, train_indices, distance_type, sample_size, seed=seed
                    )
                    coverage, accuracy, _ = search_and_compare_labels(
                        vectors, labels, test_indices, selected_indices, index, metric, limit
                    )
                    accuracy_results[sample_size][limit].append(accuracy)
                    coverage_results[sample_size][limit].append(coverage)
                    
    return coverage_results, accuracy_results

def extract_results_per_class(vectors, labels, train_indices, test_indices, distance_type, metric, samples, limits):
    per_class_coverage_results = {li: {sample: {} for sample in samples} for li in limits}
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
                if label not in per_class_coverage_results[limit][sample_size]:
                    per_class_coverage_results[limit][sample_size][label] = []
                    per_class_accuracy_results[limit][sample_size][label] = []

                per_class_coverage_results[limit][sample_size][label].append(metrics["coverage_percentage"])
                per_class_accuracy_results[limit][sample_size][label].append(metrics["match_percentage"])

    return per_class_coverage_results, per_class_accuracy_results

def extract_results_per_class_splits_avg(vectors, labels, train_test_splits, distance_type, metric, samples, limits):
    per_class_coverage_results = {li: {sample: {} for sample in samples} for li in limits}
    per_class_accuracy_results = {li: {sample: {} for sample in samples} for li in limits}

    for train_indices, test_indices in train_test_splits:
        for seed in range(40, 50):
            for sample_size in samples:
                for limit in limits:
                    index, selected_indices = build_faiss_index(
                        vectors, labels, train_indices, distance_type, sample_size, seed=seed
                    )
                    per_class_metrics = search_and_compare_labels_per_class(
                        vectors, labels, test_indices, selected_indices, index, metric, limit
                    )
                    for label, metrics in per_class_metrics.items():
                        if label not in per_class_coverage_results[limit][sample_size]:
                            per_class_coverage_results[limit][sample_size][label] = []
                            per_class_accuracy_results[limit][sample_size][label] = []

                        per_class_coverage_results[limit][sample_size][label].append(metrics["coverage_percentage"])
                        per_class_accuracy_results[limit][sample_size][label].append(metrics["match_percentage"])

    averaged_coverage_results = {li: {sample: {} for sample in samples} for li in limits}
    averaged_accuracy_results = {li: {sample: {} for sample in samples} for li in limits}

    for limit in limits:
        for sample_size in samples:
            for label in per_class_coverage_results[limit][sample_size]:
                averaged_coverage_results[limit][sample_size][label] = [
                    np.mean(per_class_coverage_results[limit][sample_size][label])
                ]
                averaged_accuracy_results[limit][sample_size][label] = [
                    np.mean(per_class_accuracy_results[limit][sample_size][label])
                ]

    return averaged_coverage_results, averaged_accuracy_results

def extract_results_limits_per_class(vectors, labels, train_indices, test_indices, distance_type, metric, samples, percentiles):
    per_class_coverage_results = {percentile: {sample: {} for sample in samples} for percentile in percentiles}
    per_class_accuracy_results = {percentile: {sample: {} for sample in samples} for percentile in percentiles}
    per_class_missed_results = {percentile: {sample: {} for sample in samples} for percentile in percentiles}

    for sample_size in samples:
        class_limits = calculate_limits_per_class(
            vectors, labels, train_indices, distance_type, sample_size, percentiles
        )
        for percentile, limits in class_limits.items():
            index, selected_indices = build_faiss_index(
                vectors, labels, train_indices, distance_type, sample_size
            )
            per_class_metrics = search_and_compare_labels_limits_per_class(
                vectors, labels, test_indices, selected_indices, index, metric, limits
            )

            for label, metrics in per_class_metrics.items():
                if label not in per_class_coverage_results[percentile][sample_size]:
                    per_class_coverage_results[percentile][sample_size][label] = []
                    per_class_accuracy_results[percentile][sample_size][label] = []
                    per_class_missed_results[percentile][sample_size][label] = []   

                per_class_coverage_results[percentile][sample_size][label].append(metrics["coverage_percentage"])
                per_class_accuracy_results[percentile][sample_size][label].append(metrics["match_percentage"])
                per_class_missed_results[percentile][sample_size][label].append(metrics["missed_percentage"])

    return per_class_coverage_results, per_class_accuracy_results, per_class_missed_results

def extract_results_limits_per_class_splits_avg(vectors, labels, train_test_splits, distance_type, metric, samples, percentiles):
    per_class_coverage_results = {percentile: {sample: {} for sample in samples} for percentile in percentiles}
    per_class_accuracy_results = {percentile: {sample: {} for sample in samples} for percentile in percentiles}
    per_class_missed_results = {percentile: {sample: {} for sample in samples} for percentile in percentiles}

    for train_indices, test_indices in train_test_splits:
        for seed in range(40, 50):
            for sample_size in samples:
                class_limits = calculate_limits_per_class(
                    vectors, labels, train_indices, distance_type, sample_size, percentiles
                )
                for percentile, limits in class_limits.items():
                    index, selected_indices = build_faiss_index(
                        vectors, labels, train_indices, distance_type, sample_size, seed=seed
                    )
                    per_class_metrics = search_and_compare_labels_limits_per_class(
                        vectors, labels, test_indices, selected_indices, index, metric, limits
                    )

                    for label, metrics in per_class_metrics.items():
                        if label not in per_class_coverage_results[percentile][sample_size]:
                            per_class_coverage_results[percentile][sample_size][label] = []
                            per_class_accuracy_results[percentile][sample_size][label] = []
                            per_class_missed_results[percentile][sample_size][label] = []  

                        per_class_coverage_results[percentile][sample_size][label].append(metrics["coverage_percentage"])
                        per_class_accuracy_results[percentile][sample_size][label].append(metrics["match_percentage"])
                        per_class_missed_results[percentile][sample_size][label].append(metrics["missed_percentage"])

    averaged_coverage_results = {percentile: {sample: {} for sample in samples} for percentile in class_limits.keys()}
    averaged_accuracy_results = {percentile: {sample: {} for sample in samples} for percentile in class_limits.keys()}
    averaged_missed_results = {percentile: {sample: {} for sample in samples} for percentile in class_limits.keys()}

    for percentile in class_limits.keys():
        for sample_size in samples:
            for label in per_class_coverage_results[percentile][sample_size]:
                averaged_coverage_results[percentile][sample_size][label] = [
                    np.mean(per_class_coverage_results[percentile][sample_size][label])
                ]
                averaged_accuracy_results[percentile][sample_size][label] = [
                    np.mean(per_class_accuracy_results[percentile][sample_size][label])
                ]
                averaged_missed_results[percentile][sample_size][label] = [
                    np.mean(per_class_missed_results[percentile][sample_size][label])
                ]

    return averaged_coverage_results, averaged_accuracy_results, averaged_missed_results

def extract_results_exclude(vectors, labels, train_indices, test_indices, distance_type, metric, samples, limits, num_classes=20):
    coverage_results_all = {}
    accuracy_results_all = {}

    for excluded_class in range(num_classes):
        test_indices_excluded_class = test_indices[np.where(labels[test_indices] == excluded_class)[0]]
        coverage_results, accuracy_results, _ = extract_results(
            vectors, labels, train_indices, test_indices_excluded_class,
            distance_type, metric, samples, limits, exclude_class=excluded_class
        )

        coverage_results_all[excluded_class] = coverage_results
        accuracy_results_all[excluded_class] = accuracy_results

    return coverage_results_all, accuracy_results_all

def plot_results(coverage_results, accuracy_results, samples, distances, reverse=False):
    num_samples = len(samples)
    fig, axes = plt.subplots(num_samples, 1, figsize=(6, 4 * num_samples), sharex=True)
    axes = axes.flatten() if num_samples > 1 else [axes]

    for i, sample_size in enumerate(samples):
        ax = axes[i]

        coverage = [coverage_results[distance][i] for distance in distances]
        accuracy = [accuracy_results[distance][i] for distance in distances]

        ax.plot(distances, accuracy, label='Accuracy', color='blue')
        ax.set_xlabel('Distance')
        ax.set_ylabel('Accuracy (%)', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax.set_ylim(0, 100)
        ax.set_title(f'Sample Size: {sample_size}')
        ax.grid(True)

        if reverse:
            ax.invert_xaxis()
            ax.set_xlabel('Similarity')

        ax2 = ax.twinx()
        ax2.plot(distances, coverage, label='Coverage %', color='green', linestyle='--')
        ax2.set_ylabel('Coverage (%)', color='green')
        ax2.set_ylim(0, 100)
        ax2.tick_params(axis='y', labelcolor='green')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def plot_results_splits(coverage_results, accuracy_results, samples, distances, reverse=False):
    num_samples = len(samples)
    fig, axes = plt.subplots(num_samples, 1, figsize=(6, 4 * num_samples), sharex=True)
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for i, sample_size in enumerate(samples):
        ax = axes[i]
        ax2 = ax.twinx()
        for run_idx in range(len(accuracy_results[sample_size][distances[0]])):
            accuracy = [accuracy_results[sample_size][distance][run_idx] for distance in distances]
            coverage = [coverage_results[sample_size][distance][run_idx] for distance in distances]
            ax.plot(distances, accuracy, color='blue', alpha=0.3, linewidth=0.5, label='Accuracy' if run_idx == 0 else None)
            ax2.plot(distances, coverage, color='green', alpha=0.3, linestyle='--', linewidth=0.5, label='Coverage %' if run_idx == 0 else None)
    
        ax.set_xlabel('Distance')
        ax.set_ylabel('Accuracy (%)', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax.set_ylim(0, 100)
        ax.set_title(f'Sample Size: {sample_size}')
        ax.grid(True)

        if reverse:
            ax.invert_xaxis()
            ax.set_xlabel('Similarity')
    
        ax2.set_ylabel('Coverage (%)', color='green')
        ax2.set_ylim(0, 100)
        ax2.tick_params(axis='y', labelcolor='green')
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def plot_results_per_class(per_class_coverage_results, per_class_accuracy_results, samples, distances, reverse=False):
    num_samples = len(samples)
    fig, axes = plt.subplots(num_samples, 1, figsize=(6, 4 * num_samples), sharex=True)
    axes = axes.flatten() if num_samples > 1 else [axes]

    unique_classes = set()
    for limit_results in per_class_coverage_results.values():
        for sample_results in limit_results.values():
            unique_classes.update(sample_results.keys())
    unique_classes = sorted(unique_classes)

    color_map = {label: plt.cm.tab20(i / 20) for i, label in enumerate(range(20))}
    real_classes = [label for label in unique_classes if label != -1]
    colors = [color_map[label] for label in real_classes]

    for i, sample_size in enumerate(samples):
        ax = axes[i]
        ax2 = ax.twinx()

        for class_idx, class_label in enumerate(real_classes):
            if class_label == -1:
                continue
            coverage = [
                per_class_coverage_results[distance][sample_size].get(class_label, [0])[0]
                for distance in distances
            ]
            accuracy = [
                per_class_accuracy_results[distance][sample_size].get(class_label, [0])[0]
                for distance in distances
            ]

            filtered_distances = [d for d, c in zip(distances, coverage) if c > 0]
            filtered_accuracy = [a for a, c in zip(accuracy, coverage) if c > 0]

            ax.plot(filtered_distances, filtered_accuracy, label=f'Class {class_label} Accuracy', color=colors[class_idx])
            ax2.plot(distances, coverage, label=f'Class {class_label} Coverage %', color=colors[class_idx], linestyle='--')
        
            if -1 in unique_classes:
                total_coverage = [
                    per_class_coverage_results[distance][sample_size].get(-1, [0])[0]
                    for distance in distances
                ]
                total_accuracy = [
                    per_class_accuracy_results[distance][sample_size].get(-1, [0])[0]
                    for distance in distances
                ]
                filtered_distances = [d for d, c in zip(distances, total_coverage) if c > 0]
                filtered_accuracy = [a for a, c in zip(total_accuracy, total_coverage) if c > 0]

                ax.plot(filtered_distances, filtered_accuracy, label='Total Accuracy', color='black', linewidth=1)
                ax2.plot(distances, total_coverage, label='Total Coverage %', color='black', linestyle='--', linewidth=1)

            ax.set_xlabel('Distance')
            ax.set_ylabel('Accuracy (%) —')
            ax.tick_params(axis='y')
            ax.set_ylim(0, 100)
            ax.set_title(f'Sample Size: {sample_size}')
            ax.grid(True)

        ax2.set_ylabel('Coverage (%) --')
        ax2.set_ylim(0, 100)
        ax2.tick_params(axis='y')

        if reverse:
            ax.invert_xaxis()
            ax.set_xlabel('Similarity')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def plot_results_limits_per_class(per_class_coverage_results, per_class_accuracy_results, per_class_missed_results, samples, percentiles, reverse=False):
    num_samples = len(samples)
    fig, axes = plt.subplots(num_samples, 1, figsize=(6, 4 * num_samples), sharex=True)
    axes = axes.flatten() if num_samples > 1 else [axes]

    unique_classes = set()
    for limit_results in per_class_coverage_results.values():
        for sample_results in limit_results.values():
            unique_classes.update(sample_results.keys())
    unique_classes = sorted(unique_classes)

    color_map = {label: plt.cm.tab20(i / 20) for i, label in enumerate(range(20))}
    real_classes = [label for label in unique_classes if label != -1]
    colors = [color_map[label] for label in real_classes]

    for i, sample_size in enumerate(samples):
        ax = axes[i]
        ax2 = ax.twinx()

        for class_idx, class_label in enumerate(real_classes):
            if class_label == -1:
                continue
            coverage = [
                per_class_coverage_results[percentile][sample_size].get(class_label, [0])[0]
                for percentile in percentiles
            ]
            accuracy = [
                per_class_accuracy_results[percentile][sample_size].get(class_label, [0])[0]
                for percentile in percentiles
            ]
            missed = [
                per_class_missed_results[percentile][sample_size].get(class_label, [0])[0]
                for percentile in percentiles
            ]

            filtered_percentiles = [d for d, c in zip(percentiles, coverage) if c > 0]
            filtered_accuracy = [a for a, c in zip(accuracy, coverage) if c > 0]

            ax.plot(filtered_percentiles, filtered_accuracy, label=f'Class {class_label} Accuracy', color=colors[class_idx])
            ax.set_xlabel('Percentile Limit')
            ax.set_ylabel('Accuracy (%) —')
            ax.tick_params(axis='y')
            ax.set_ylim(0, 100)
            ax.set_title(f'Sample Size: {sample_size}')
            ax.grid(True)
            
            ax2.plot(percentiles, coverage, label=f'Class {class_label} Coverage %', color=colors[class_idx], linestyle='--')
            ax2.plot(percentiles, missed, label=f'Class {class_label} Missed %', color=colors[class_idx], linestyle=':')
        
            if -1 in unique_classes:
                total_coverage = [
                    per_class_coverage_results[percentile][sample_size].get(-1, [0])[0]
                    for percentile in percentiles
                ]
                total_accuracy = [
                    per_class_accuracy_results[percentile][sample_size].get(-1, [0])[0]
                    for percentile in percentiles
                ]
                total_missed = [
                    per_class_missed_results[percentile][sample_size].get(-1, [0])[0]
                    for percentile in percentiles
                ]
                filtered_percentiles = [d for d, c in zip(percentiles, total_coverage) if c > 0]
                filtered_accuracy = [a for a, c in zip(total_accuracy, total_coverage) if c > 0]

                ax.plot(filtered_percentiles, filtered_accuracy, label='Total Accuracy', color='black', linewidth=1)
                ax2.plot(percentiles, total_coverage, label='Total Coverage %', color='black', linestyle='--', linewidth=1)
                ax2.plot(percentiles, total_missed,  label='Total Missed %', color='black', linestyle=':')

        ax2.set_ylabel('Coverage (%) --')
        ax2.set_ylim(0, 100)
        ax2.tick_params(axis='y')

        if reverse:
            ax.invert_xaxis()
            ax.set_xlabel('Similarity')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def plot_results_per_excluded_class(coverage_results_all, accuracy_results_all, samples, limits, reverse=False):
    num_samples = len(samples)
    fig, axes = plt.subplots(num_samples, 1, figsize=(6, 4 * num_samples), sharex=True)

    axes = axes.flatten() if num_samples > 1 else [axes]

    excluded_classes = list(coverage_results_all.keys())
    color_map = {label: plt.cm.tab20(i / 20) for i, label in enumerate(range(20))}
    colors = [color_map[label] for label in excluded_classes]

    for i, sample_size in enumerate(samples):
        ax = axes[i]
        ax2 = ax.twinx()

        for class_idx, excluded_class in enumerate(excluded_classes):
            coverage_results = coverage_results_all[excluded_class]
            accuracy_results = accuracy_results_all[excluded_class]

            accuracies = [np.mean(accuracy_results[limit][i]) for limit in limits]
            coverage = [np.mean(coverage_results[limit][i]) for limit in limits]

            ax.plot(limits, accuracies, linestyle='-', color=colors[class_idx])
            ax2.plot(limits, coverage, linestyle='--', color=colors[class_idx])

        ax.set_xlabel('Distance')
        ax.set_ylabel('Accuracy (%) —')
        ax.tick_params(axis='y')
        ax.set_ylim(0, 100)
        ax.set_title(f'Sample Size: {sample_size}')
        ax.grid(True)
    
        ax2.set_ylabel('Coverage (%) --')
        ax2.tick_params(axis='y')
        ax2.set_ylim(0, 100)

        if reverse:
            ax.invert_xaxis()
            ax.set_xlabel('Similarity')

    plt.tight_layout()
    plt.show()