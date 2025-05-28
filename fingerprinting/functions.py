import numpy as np
import matplotlib.pyplot as plt
import faiss

vectors_baseline = np.load('/home/ev357/tcbench/src/fingerprinting/mirage19/baseline_vectors.npy')
labels_baseline = np.load('/home/ev357/tcbench/src/fingerprinting/mirage19/baseline_labels.npy')
vectors_embeddings = np.load('/home/ev357/tcbench/src/fingerprinting/mirage19/embeddings_vectors.npy')
labels_embeddings = np.load('/home/ev357/tcbench/src/fingerprinting/mirage19/embeddings_labels.npy')

def build_faiss(data_type, distance_type, samples_per_class=None, exclude_class=None, seed=42, train_indices=None):
    if data_type == 'baseline':
        data = vectors_baseline
    elif data_type == 'embeddings':
        data = vectors_embeddings

    if train_indices is None:
        selected_indices = select_indices_by_class(
            data_type, samples_per_class, exclude_class=exclude_class, seed=seed
        )
    else:
        selected_indices = train_indices

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

def select_indices_by_class(data_type, samples_per_class, exclude_class=None, seed=42):
    np.random.seed(seed)
    if data_type == 'baseline':
        labels = labels_baseline
    elif data_type == 'embeddings':    
        labels = labels_embeddings
    
    unique_labels = np.unique(labels)
    selected_indices = []
    for label in unique_labels:
        if exclude_class is not None and label == exclude_class:
            continue
        label_indices = np.where(labels == label)[0]
        if samples_per_class is None:
            n_select = int(np.floor(len(label_indices) * 0.9))
            selected_label_indices = np.random.choice(
                label_indices, size=n_select, replace=False
            )
        else:
            selected_label_indices = np.random.choice(
                label_indices, size=min(samples_per_class, len(label_indices)), replace=False
            )
        selected_indices.extend(selected_label_indices)
    return np.array(selected_indices)

def search_faiss(data_type, selected_indices, index, metric, limit=None):
    if data_type == 'baseline':
        data = vectors_baseline
        labels = labels_baseline
    elif data_type == 'embeddings':
        data = vectors_embeddings
        labels = labels_embeddings
    
    all_indices = np.arange(len(labels))
    test_indices = np.setdiff1d(all_indices, selected_indices)

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
    preds = []
    trues = []

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
        preds.append(neighbor_labels[i, 0])
        trues.append(test_labels[i])

    coverage_percentage = (coverage / len(test_indices)) * 100
    match_percentage = (matches / coverage) * 100 if coverage > 0 else 0

    return coverage_percentage, match_percentage, min_distances, preds, trues

def search_faiss_majority(data_type, selected_indices, index, metric, limit=None):
    if data_type == 'baseline':
        data = vectors_baseline
        labels = labels_baseline
    elif data_type == 'embeddings':
        data = vectors_embeddings
        labels = labels_embeddings
    
    all_indices = np.arange(len(labels))
    test_indices = np.setdiff1d(all_indices, selected_indices)

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

def search_faiss_per_class(data_type, selected_indices, index, metric, limit=None):
    if data_type == 'baseline':
        data = vectors_baseline
        labels = labels_baseline
    elif data_type == 'embeddings':
        data = vectors_embeddings
        labels = labels_embeddings
    
    all_indices = np.arange(len(labels))
    test_indices = np.setdiff1d(all_indices, selected_indices)
    
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

def search_faiss_limits_per_class(data_type, selected_indices, index, metric, class_limits):
    if data_type == 'baseline':
        data = vectors_baseline
        labels = labels_baseline
    elif data_type == 'embeddings':
        data = vectors_embeddings
        labels = labels_embeddings
    
    all_indices = np.arange(len(labels))
    test_indices = np.setdiff1d(all_indices, selected_indices)

    unique_labels = np.unique(labels)
    per_class_preds = {label: [] for label in unique_labels}
    per_class_trues = {label: [] for label in unique_labels}
    per_class_preds[-1] = []
    per_class_trues[-1] = []
    
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
                per_class_preds[test_class].append(nn_class)
                per_class_trues[test_class].append(test_class)
                per_class_preds[-1].append(nn_class)
                per_class_trues[-1].append(test_class)
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
                per_class_preds[test_class].append(nn_class)
                per_class_trues[test_class].append(test_class)
                per_class_preds[-1].append(nn_class)
                per_class_trues[-1].append(test_class)
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
    
    return per_class_metrics, per_class_preds, per_class_trues

def search_faiss_limits_per_class_iterative(data_type, selected_indices, index, metric, class_limits):
    if data_type == 'baseline':
        data = vectors_baseline
        labels = labels_baseline
    elif data_type == 'embeddings':
        data = vectors_embeddings
        labels = labels_embeddings

    all_indices = np.arange(len(labels))
    test_indices = np.setdiff1d(all_indices, selected_indices)

    if metric == "distance":
        max_limit = max(class_limits.values())
        lims, D, I = index.range_search(data[test_indices], max_limit**2)
    elif metric == "similarity":
        min_limit = min(class_limits.values())
        query_vectors = data[test_indices]
        query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
        lims, D, I = index.range_search(query_vectors, min_limit)

    sorted_D = np.zeros_like(D)
    sorted_I = np.zeros_like(I)

    for i in range(len(lims) - 1):
        start, end = lims[i], lims[i + 1]
        distances = D[start:end]
        indices = I[start:end]
        sorted_order = np.argsort(distances) if metric == "distance" else np.argsort(distances)[::-1]
        sorted_D[start:end] = distances[sorted_order]
        sorted_I[start:end] = indices[sorted_order]

    test_labels = labels[test_indices]
    unique_labels = np.unique(test_labels)
    per_class_results = {label: {"coverage": 0, "matches": 0, "total": 0} for label in unique_labels}
    per_class_results[-1] = {"coverage": 0, "matches": 0, "total": 0}

    per_class_preds = {label: [] for label in unique_labels}
    per_class_trues = {label: [] for label in unique_labels}
    per_class_preds[-1] = []
    per_class_trues[-1] = []

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

        dists_or_sims = neighbors_distances
        classes = neighbors_classes

        while len(dists_or_sims) > 0:
            score = dists_or_sims[0]
            nn_class = classes[0]
            nn_limit = class_limits.get(nn_class, None)

            match_condition = (
                (metric == "distance" and np.sqrt(score) <= nn_limit) or
                (metric == "similarity" and score >= nn_limit)
            )

            if match_condition:
                per_class_results[test_class]["coverage"] += 1
                per_class_results[-1]["coverage"] += 1

                per_class_preds[test_class].append(nn_class)
                per_class_trues[test_class].append(test_class)
                per_class_preds[-1].append(nn_class)
                per_class_trues[-1].append(test_class)

                if nn_class == test_class:
                    per_class_results[test_class]["matches"] += 1
                    per_class_results[-1]["matches"] += 1
                break
            else:
                mask = classes != nn_class
                dists_or_sims = dists_or_sims[mask]
                classes = classes[mask]

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

    return per_class_metrics, per_class_preds, per_class_trues

def calculate_limits(data_type, distance_type, samples_per_class, percentiles, train_indices=None, seed=42,):
    if data_type == 'baseline':
        data = vectors_baseline
        labels = labels_baseline
    elif data_type == 'embeddings':
        data = vectors_embeddings
        labels = labels_embeddings

    if train_indices is None:
        selected_indices = select_indices_by_class(
            data_type, samples_per_class, exclude_class=None, seed=seed 
        )
    else:
        selected_indices = train_indices
    
    selected_labels = labels[selected_indices]
    unique_labels = np.unique(selected_labels)
    all_nn_distances = []

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

        all_nn_distances.extend(nn_distances)
    
    all_nn_distances = np.array(all_nn_distances)

    limits_per_percentile = {}
    for percentile in percentiles:
        if len(all_nn_distances) == 0:
            limit = 0
        else:
            if distance_type == 'euclidean':
                limit = np.percentile(all_nn_distances, percentile)
            elif distance_type == 'cosine':
                limit = np.percentile(all_nn_distances, 100 - percentile)
        limits_per_percentile[percentile] = limit
    return limits_per_percentile

def calculate_limits_per_class(data_type, distance_type, samples_per_class, percentiles, train_indices=None, seed=42):
    if data_type == 'baseline':
        data = vectors_baseline
        labels = labels_baseline
    elif data_type == 'embeddings':
        data = vectors_embeddings
        labels = labels_embeddings

    if train_indices is not None:
        selected_indices = np.array(train_indices)
    else:
        selected_indices = select_indices_by_class(
            data_type, samples_per_class, exclude_class=None, seed=seed 
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
                if distance_type == 'euclidean':
                    limits[label] = np.percentile(distances, percentile)
                elif distance_type == 'cosine':
                    limits[label] = np.percentile(distances, 100 - percentile)
        limits_per_percentile[percentile] = limits
    return limits_per_percentile

def extract_results(data_type, distance_type, metric, samples_per_class, limits, exclude_class=None):
    coverage_results = {li: [] for li in limits}
    accuracy_results = {li: [] for li in limits}

    for limit in limits:
        for sample_size in samples_per_class:
            index, selected_indices = build_faiss(
                data_type, distance_type, sample_size, exclude_class=exclude_class
            )
            coverage, accuracy, min_distances = search_faiss(
                data_type, selected_indices, index, metric, limit
            )
            coverage_results[limit].append(coverage)
            accuracy_results[limit].append(accuracy)   
    return coverage_results, accuracy_results, min_distances

def extract_results_splits(data_type, distance_type, metric, samples, limits):
    accuracy_results = {sample_size: {limit: [] for limit in limits} for sample_size in samples}
    coverage_results = {sample_size: {limit: [] for limit in limits} for sample_size in samples}
    all_preds = {sample_size: {limit: [] for limit in limits} for sample_size in samples}
    all_trues = {sample_size: {limit: [] for limit in limits} for sample_size in samples}

    for seed in range(50):
        for sample_size in samples:
            for limit in limits:
                index, selected_indices = build_faiss(
                    data_type, distance_type, sample_size, seed=seed
                )
                coverage, accuracy, _, preds, trues = search_faiss(
                    data_type, selected_indices, index, metric, limit
                )
                accuracy_results[sample_size][limit].append(accuracy)
                coverage_results[sample_size][limit].append(coverage)
                all_preds[sample_size][limit].extend(preds)
                all_trues[sample_size][limit].extend(trues)
                    
    return coverage_results, accuracy_results, all_preds, all_trues

def extract_results_majority(data_type, distance_type, metric, samples_per_class, limits):
    coverage_results = {li: [] for li in limits}
    accuracy_results = {li: [] for li in limits}

    for limit in limits:
        for sample_size in samples_per_class:
            index, selected_indices = build_faiss(
                data_type, distance_type, sample_size
            )
            coverage, accuracy = search_faiss_majority(
                data_type, selected_indices, index, metric, limit
            )
            coverage_results[limit].append(coverage)
            accuracy_results[limit].append(accuracy)   
    return coverage_results, accuracy_results

def extract_results_limits(data_type, distance_type, metric, samples_per_class, percentiles):
    accuracy_results = {p: [] for p in percentiles}
    coverage_results = {p: [] for p in percentiles}

    for sample_size in samples_per_class:
        limits = calculate_limits(
            data_type, distance_type, sample_size, percentiles
        )
        for percentile, limit in limits.items(): 
            index, selected_indices = build_faiss(
                data_type, distance_type, sample_size
            )
            coverage, accuracy, _ = search_faiss(
                data_type, selected_indices, index, metric, limit
            )
            coverage_results[percentile].append(coverage)
            accuracy_results[percentile].append(accuracy)   
    return coverage_results, accuracy_results

def extract_results_limits_splits(data_type, distance_type, metric, samples, percentiles):
    accuracy_results = {sample_size: {p: [] for p in percentiles} for sample_size in samples}
    coverage_results = {sample_size: {p: [] for p in percentiles} for sample_size in samples}
    all_preds = {sample_size: {p: [] for p in percentiles} for sample_size in samples}
    all_trues = {sample_size: {p: [] for p in percentiles} for sample_size in samples}

    for seed in range(50):
        for sample_size in samples:
            limits = calculate_limits(
                data_type, distance_type, sample_size, percentiles
            )
            for percentile, limit in limits.items():
                index, selected_indices = build_faiss(
                    data_type, distance_type, sample_size, seed=seed
                )
                coverage, accuracy, _, preds, trues = search_faiss(
                    data_type, selected_indices, index, metric, limit
                )
                accuracy_results[sample_size][percentile].append(accuracy)
                coverage_results[sample_size][percentile].append(coverage)
                all_preds[sample_size][percentile].extend(preds)
                all_trues[sample_size][percentile].extend(trues)

    return coverage_results, accuracy_results, all_preds, all_trues

def extract_results_per_class(data_type, distance_type, metric, samples, limits):
    per_class_coverage_results = {li: {sample: {} for sample in samples} for li in limits}
    per_class_accuracy_results = {li: {sample: {} for sample in samples} for li in limits}

    for limit in limits:
        for sample_size in samples:
            index, selected_indices = build_faiss(
                data_type, distance_type, sample_size
            )
            per_class_metrics = search_faiss_per_class(
                data_type, selected_indices, index, metric, limit
            )

            for label, metrics in per_class_metrics.items():
                if label not in per_class_coverage_results[limit][sample_size]:
                    per_class_coverage_results[limit][sample_size][label] = []
                    per_class_accuracy_results[limit][sample_size][label] = []

                per_class_coverage_results[limit][sample_size][label].append(metrics["coverage_percentage"])
                per_class_accuracy_results[limit][sample_size][label].append(metrics["match_percentage"])

    return per_class_coverage_results, per_class_accuracy_results

def extract_results_per_class_splits_avg(data_type, distance_type, metric, samples, limits):
    per_class_coverage_results = {li: {sample: {} for sample in samples} for li in limits}
    per_class_accuracy_results = {li: {sample: {} for sample in samples} for li in limits}

    for seed in range(50):
        for sample_size in samples:
            for limit in limits:
                index, selected_indices = build_faiss(
                    data_type, distance_type, sample_size, seed=seed
                )
                per_class_metrics = search_faiss_per_class(
                    data_type, selected_indices, index, metric, limit
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

def extract_results_limits_per_class(data_type, distance_type, metric, samples, percentiles):
    per_class_coverage_results = {percentile: {sample: {} for sample in samples} for percentile in percentiles}
    per_class_accuracy_results = {percentile: {sample: {} for sample in samples} for percentile in percentiles}
    per_class_missed_results = {percentile: {sample: {} for sample in samples} for percentile in percentiles}

    for sample_size in samples:
        class_limits = calculate_limits_per_class(
            data_type, distance_type, sample_size, percentiles
        )
        for percentile, limits in class_limits.items():
            index, selected_indices = build_faiss(
                data_type, distance_type, sample_size
            )
            per_class_metrics = search_faiss_limits_per_class(
                data_type, selected_indices, index, metric, limits
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

def extract_results_limits_per_class_splits_avg(data_type, distance_type, metric, samples, percentiles):
    per_class_coverage_results = {percentile: {sample: {} for sample in samples} for percentile in percentiles}
    per_class_accuracy_results = {percentile: {sample: {} for sample in samples} for percentile in percentiles}
    per_class_missed_results = {percentile: {sample: {} for sample in samples} for percentile in percentiles}
    per_class_preds_all = {percentile: {sample: {} for sample in samples} for percentile in percentiles}
    per_class_trues_all = {percentile: {sample: {} for sample in samples} for percentile in percentiles}
    
    for seed in range(50):
        for sample_size in samples:
            class_limits = calculate_limits_per_class(
                data_type, distance_type, sample_size, percentiles
            )
            for percentile, limits in class_limits.items():
                index, selected_indices = build_faiss(
                    data_type, distance_type, sample_size, seed=seed
                )
                per_class_metrics, per_class_preds, per_class_trues = search_faiss_limits_per_class(
                    data_type, selected_indices, index, metric, limits
                )

                for label in per_class_preds:
                    if label not in per_class_preds_all[percentile][sample_size]:
                        per_class_preds_all[percentile][sample_size][label] = []
                        per_class_trues_all[percentile][sample_size][label] = []
                    per_class_preds_all[percentile][sample_size][label].extend(per_class_preds[label])
                    per_class_trues_all[percentile][sample_size][label].extend(per_class_trues[label])

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

    return averaged_coverage_results, averaged_accuracy_results, averaged_missed_results, per_class_preds_all, per_class_trues_all

def extract_results_limits_per_class_iterative(data_type, distance_type, metric, samples, percentiles):
    per_class_coverage_results = {percentile: {sample: {} for sample in samples} for percentile in percentiles}
    per_class_accuracy_results = {percentile: {sample: {} for sample in samples} for percentile in percentiles}

    for sample_size in samples:
        class_limits = calculate_limits_per_class(
            data_type, distance_type, sample_size, percentiles
        )
        for percentile, limits in class_limits.items():
            index, selected_indices = build_faiss(
                data_type, distance_type, sample_size
            )
            per_class_metrics = search_faiss_limits_per_class_iterative(
                data_type, selected_indices, index, metric, limits
            )

            for label, metrics in per_class_metrics.items():
                if label not in per_class_coverage_results[percentile][sample_size]:
                    per_class_coverage_results[percentile][sample_size][label] = []
                    per_class_accuracy_results[percentile][sample_size][label] = [] 

                per_class_coverage_results[percentile][sample_size][label].append(metrics["coverage_percentage"])
                per_class_accuracy_results[percentile][sample_size][label].append(metrics["match_percentage"])

    return per_class_coverage_results, per_class_accuracy_results

def extract_results_limits_per_class_iterative_splits_avg(data_type, distance_type, metric, samples, percentiles):
    per_class_coverage_results = {percentile: {sample: {} for sample in samples} for percentile in percentiles}
    per_class_accuracy_results = {percentile: {sample: {} for sample in samples} for percentile in percentiles}
    per_class_preds_all = {percentile: {sample: {} for sample in samples} for percentile in percentiles}
    per_class_trues_all = {percentile: {sample: {} for sample in samples} for percentile in percentiles}


    for seed in range(50):
        for sample_size in samples:
            class_limits = calculate_limits_per_class(
                data_type, distance_type, sample_size, percentiles
            )
            for percentile, limits in class_limits.items():
                index, selected_indices = build_faiss(
                    data_type, distance_type, sample_size, seed=seed
                )
                per_class_metrics, per_class_preds, per_class_trues = search_faiss_limits_per_class_iterative(
                    data_type, selected_indices, index, metric, limits
                )

                for label in per_class_preds:
                    if label not in per_class_preds_all[percentile][sample_size]:
                        per_class_preds_all[percentile][sample_size][label] = []
                        per_class_trues_all[percentile][sample_size][label] = []
                    per_class_preds_all[percentile][sample_size][label].extend(per_class_preds[label])
                    per_class_trues_all[percentile][sample_size][label].extend(per_class_trues[label])

                for label, metrics in per_class_metrics.items():
                    if label not in per_class_coverage_results[percentile][sample_size]:
                        per_class_coverage_results[percentile][sample_size][label] = []
                        per_class_accuracy_results[percentile][sample_size][label] = []

                    per_class_coverage_results[percentile][sample_size][label].append(metrics["coverage_percentage"])
                    per_class_accuracy_results[percentile][sample_size][label].append(metrics["match_percentage"])

    averaged_coverage_results = {percentile: {sample: {} for sample in samples} for percentile in class_limits.keys()}
    averaged_accuracy_results = {percentile: {sample: {} for sample in samples} for percentile in class_limits.keys()}

    for percentile in class_limits.keys():
        for sample_size in samples:
            for label in per_class_coverage_results[percentile][sample_size]:
                averaged_coverage_results[percentile][sample_size][label] = [
                    np.mean(per_class_coverage_results[percentile][sample_size][label])
                ]
                averaged_accuracy_results[percentile][sample_size][label] = [
                    np.mean(per_class_accuracy_results[percentile][sample_size][label])
                ]

    return averaged_coverage_results, averaged_accuracy_results, per_class_preds_all, per_class_trues_all

def plot_results(coverage_results, accuracy_results, samples, distances, name, reverse=False, percentile=False):
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
        
        if percentile:
            ax.set_xlabel('Percentile')

        ax2 = ax.twinx()
        ax2.plot(distances, coverage, label='Coverage %', color='green', linestyle='--')
        ax2.set_ylabel('Coverage (%)', color='green')
        ax2.set_ylim(0, 100)
        ax2.tick_params(axis='y', labelcolor='green')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f"/home/ev357/tcbench/src/fingerprinting/thresholds/{name}.png")

def plot_results_splits(coverage_results, accuracy_results, samples, distances, name, reverse=False, percentile=False):
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
        
        if percentile:
            ax.set_xlabel('Percentile')
    
        ax2.set_ylabel('Coverage (%)', color='green')
        ax2.set_ylim(0, 100)
        ax2.tick_params(axis='y', labelcolor='green')
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f"/home/ev357/tcbench/src/fingerprinting/thresholds/{name}.png")

def plot_results_per_class(per_class_coverage_results, per_class_accuracy_results, samples, distances, name, reverse=False, percentile=False):
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
        
        if percentile:
            ax.set_xlabel('Percentile')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f"/home/ev357/tcbench/src/fingerprinting/thresholds/{name}.png")

def plot_results_per_class_missed(per_class_coverage_results, per_class_accuracy_results, per_class_missed_results, samples, percentiles, reverse=False):
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
            ax.set_xlabel('Percentile')
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

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def select_next_batch(unlabeled_indices, batch_size, seed=42):
    unlabeled_indices = list(unlabeled_indices)
    np.random.seed(seed)
    if len(unlabeled_indices) <= batch_size:
        batch = unlabeled_indices
    else:
        batch = np.random.choice(unlabeled_indices, size=batch_size, replace=False)
    return list(batch)

def semi_supervised(data_type, distance_type, metric, starting_samples_per_class, high_confidence_percentile, low_confidence_percentile, batch_size=1024, seed=42):
    if data_type == 'baseline':
        data = vectors_baseline
        labels = labels_baseline
    elif data_type == 'embeddings':
        data = vectors_embeddings
        labels = labels_embeddings
    
    all_indices = np.arange(len(labels))
    labeled_indices = select_indices_by_class(data_type, starting_samples_per_class, seed=seed)
    labeled_labels = labels[labeled_indices].copy()
    original_labeled_indices = set(labeled_indices)
    unlabeled_indices = set(all_indices) - set(labeled_indices)

    batch_accuracies = []
    batch_coverages = []
    batch_training_points = []
    batch_correct_training_points = []

    training_points = len(labeled_indices)
    correct_training_points = training_points
    batch_training_points.append(training_points)
    batch_correct_training_points.append(correct_training_points)

    while unlabeled_indices:
        index, _ = build_faiss(data_type, distance_type, seed=seed, train_indices=labeled_indices)
        batch_indices = select_next_batch(unlabeled_indices, batch_size=batch_size, seed=seed)
        batch_vectors = data[batch_indices]
        if metric == 'distance':
            D, I = index.search(batch_vectors, k=1)
        elif metric == 'similarity':
            batch_vectors = batch_vectors / np.linalg.norm(batch_vectors, axis=1, keepdims=True)
            D, I = index.search(batch_vectors, k=1)
        
        batch_pred_labels = []
        batch_true_labels = []
        covered = []

        limits = calculate_limits(
            data_type, distance_type, None, [high_confidence_percentile, low_confidence_percentile], train_indices=labeled_indices, seed=seed
        )
        high_confidence_threshold = limits[high_confidence_percentile]
        low_confidence_threshold = limits[low_confidence_percentile]

        for i, (idx, dist, nn_idx) in enumerate(zip(batch_indices, D, I)):
            nn_label = labeled_labels[nn_idx[0]]
            true_label = labels[idx]
            nn_global_idx = labeled_indices[nn_idx[0]]

            if metric == 'distance':
                dist = np.sqrt(dist)
                if dist < high_confidence_threshold:
                    if not np.isclose(dist, 0) and nn_global_idx in original_labeled_indices:
                        labeled_indices = np.append(labeled_indices, idx)
                        labeled_labels = np.append(labeled_labels, nn_label)
                        training_points += 1
                        if nn_label == true_label:
                            correct_training_points += 1
                    batch_pred_labels.append(nn_label)
                    batch_true_labels.append(true_label)
                    covered.append(True)
                elif dist < low_confidence_threshold:
                    batch_pred_labels.append(nn_label)
                    batch_true_labels.append(true_label)
                    covered.append(True)
                else:
                    covered.append(False)
            elif metric == 'similarity':
                if dist > high_confidence_threshold:
                    if not np.isclose(dist, 1.0) and nn_global_idx in original_labeled_indices:
                        labeled_indices = np.append(labeled_indices, idx)
                        labeled_labels = np.append(labeled_labels, nn_label)
                        training_points += 1
                        if nn_label == true_label:
                            correct_training_points += 1
                    batch_pred_labels.append(nn_label)
                    batch_true_labels.append(true_label)
                    covered.append(True)
                elif dist > low_confidence_threshold:
                    batch_pred_labels.append(nn_label)
                    batch_true_labels.append(true_label)
                    covered.append(True)
                else:
                    covered.append(False)

        num_covered = sum(covered)
        coverage = (num_covered / len(batch_indices)) * 100 if batch_indices else 0
        if num_covered > 0:
            correct = sum(
                pred == true for pred, true, cov in zip(batch_pred_labels, batch_true_labels, covered) if cov
            )
            accuracy = (correct / num_covered) * 100
        else:
            accuracy = 0

        batch_accuracies.append(accuracy)
        batch_coverages.append(coverage)
        batch_training_points.append(training_points)
        batch_correct_training_points.append(correct_training_points)
        unlabeled_indices -= set(batch_indices)

    return batch_accuracies, batch_coverages, batch_training_points, batch_correct_training_points

def semi_supervised_per_class(data_type, distance_type, metric, starting_samples_per_class, high_confidence_percentile, low_confidence_percentile, batch_size=1024, seed=42):
    if data_type == 'baseline':
        data = vectors_baseline
        labels = labels_baseline
    elif data_type == 'embeddings':
        data = vectors_embeddings
        labels = labels_embeddings

    all_indices = np.arange(len(labels))
    labeled_indices = list(select_indices_by_class(data_type, starting_samples_per_class, seed=seed))
    labeled_labels = labels[labeled_indices].copy()
    original_labeled_indices = set(labeled_indices)
    unlabeled_indices = set(all_indices) - set(labeled_indices)

    per_class_accuracies = {}
    per_class_coverages = {}
    unique_labels = np.unique(labels)
    for label in unique_labels:
        per_class_accuracies[label] = []
        per_class_coverages[label] = []
    per_class_accuracies[-1] = []
    per_class_coverages[-1] = []

    batch_training_points = []
    batch_correct_training_points = []

    training_points = len(labeled_indices)
    correct_training_points = training_points
    batch_training_points.append(training_points)
    batch_correct_training_points.append(correct_training_points)

    class_limits = calculate_limits_per_class(
        data_type, distance_type, starting_samples_per_class, [high_confidence_percentile, low_confidence_percentile], train_indices=labeled_indices, seed=seed
    )
    
    high_confidence_limits = class_limits[high_confidence_percentile]
    low_confidence_limits = class_limits[low_confidence_percentile]

    while unlabeled_indices:
        index, _ = build_faiss(data_type, distance_type, seed=seed, train_indices=labeled_indices)
        batch_indices = select_next_batch(unlabeled_indices, batch_size=batch_size, seed=seed)
        if metric == 'distance':
            batch_vectors = data[batch_indices]
            D, I = index.search(batch_vectors, k=1)
        elif metric == 'similarity':
            batch_vectors = data[batch_indices]
            batch_vectors = batch_vectors / np.linalg.norm(batch_vectors, axis=1, keepdims=True)
            D, I = index.search(batch_vectors, k=1)

        batch_pred_labels = {label: [] for label in unique_labels}
        batch_true_labels = {label: [] for label in unique_labels}
        covered = {label: [] for label in unique_labels}
        batch_pred_labels[-1] = []
        batch_true_labels[-1] = []
        covered[-1] = []

        for idx, dist, nn_idx in zip(batch_indices, D, I):
            nn_label = labeled_labels[nn_idx[0]]
            true_label = labels[idx]
            nn_global_idx = labeled_indices[nn_idx[0]]

            if metric == 'distance':
                dist = np.sqrt(dist)
                high_thr = high_confidence_limits[nn_label]
                low_thr = low_confidence_limits[nn_label]
                if dist < high_thr:
                    if not np.isclose(dist, 0) and nn_global_idx in original_labeled_indices:
                        labeled_indices = np.append(labeled_indices, idx)
                        labeled_labels = np.append(labeled_labels, nn_label)
                        training_points += 1
                        if nn_label == true_label:
                            correct_training_points += 1
                    batch_pred_labels[nn_label].append(nn_label)
                    batch_true_labels[nn_label].append(true_label)
                    covered[nn_label].append(True)
                elif dist < low_thr:
                    batch_pred_labels[nn_label].append(nn_label)
                    batch_true_labels[nn_label].append(true_label)
                    covered[nn_label].append(True)
                else:
                    covered[nn_label].append(False)
            elif metric == 'similarity':
                high_thr = high_confidence_limits[nn_label]
                low_thr = low_confidence_limits[nn_label]
                if dist > high_thr:
                    if not np.isclose(dist, 1.0) and nn_global_idx in original_labeled_indices:
                        labeled_indices = np.append(labeled_indices, idx)
                        labeled_labels = np.append(labeled_labels, nn_label)
                        training_points += 1
                        if nn_label == true_label:
                            correct_training_points += 1
                    batch_pred_labels[nn_label].append(nn_label)
                    batch_true_labels[nn_label].append(true_label)
                    covered[nn_label].append(True)
                elif dist > low_thr:
                    batch_pred_labels[nn_label].append(nn_label)
                    batch_true_labels[nn_label].append(true_label)
                    covered[nn_label].append(True)
                else:
                    covered[nn_label].append(False)

            if metric == 'distance':
                high_thr = high_confidence_limits[nn_label]
                low_thr = low_confidence_limits[nn_label]
                if dist < low_thr:
                    batch_pred_labels[-1].append(nn_label)
                    batch_true_labels[-1].append(true_label)
                    covered[-1].append(True)
                else:
                    covered[-1].append(False)
            elif metric == 'similarity':
                high_thr = high_confidence_limits[nn_label]
                low_thr = low_confidence_limits[nn_label]
                if dist > low_thr:
                    batch_pred_labels[-1].append(nn_label)
                    batch_true_labels[-1].append(true_label)
                    covered[-1].append(True)
                else:
                    covered[-1].append(False)

        for label in list(unique_labels) + [-1]:
            if label == -1:
                denom = len(batch_indices)
            else:
                denom = sum([labels[idx] == label for idx in batch_indices])
            num_covered = sum(covered[label])
            coverage = (num_covered / denom) * 100 if batch_indices else 0
            if num_covered > 0:
                correct = sum(
                    pred == true for pred, true, cov in zip(batch_pred_labels[label], batch_true_labels[label], covered[label]) if cov
                )
                accuracy = (correct / num_covered) * 100
            else:
                accuracy = 0
            per_class_accuracies[label].append(accuracy)
            per_class_coverages[label].append(coverage)

        batch_training_points.append(training_points)
        batch_correct_training_points.append(correct_training_points)
        unlabeled_indices -= set(batch_indices)

    return per_class_accuracies, per_class_coverages, batch_training_points, batch_correct_training_points

def run_multiple_avg(data_type, distance_type, metric, starting_sample_size, high_confidence_percentile, low_confidence_percentile, num_runs, batch_size=1024):
    accuracy_all = []
    coverage_all = []
    train_all = []
    cor_train_all = []
    for seed in range(1, num_runs + 1):
        acc, cov, tp, ctp = semi_supervised(data_type, distance_type, metric, starting_sample_size, high_confidence_percentile, low_confidence_percentile, batch_size=batch_size, seed=seed)
        accuracy_all.append(acc)
        coverage_all.append(cov)
        train_all.append(tp)
        cor_train_all.append(ctp)
    accuracy_all = np.array(accuracy_all)
    coverage_all = np.array(coverage_all)
    train_all = np.array(train_all)
    cor_train_all = np.array(cor_train_all)
    accuracy_avg = np.mean(accuracy_all, axis=0)
    coverage_avg = np.mean(coverage_all, axis=0)
    train_avg = np.round(np.mean(train_all, axis=0)).astype(int)
    cor_train_avg = np.round(np.mean(cor_train_all, axis=0)).astype(int)
    return coverage_avg, accuracy_avg, train_avg, cor_train_avg

def run_multiple_per_class_avg(data_type, distance_type, metric, starting_sample_size, high_confidence_percentile, low_confidence_percentile, num_runs, batch_size=1024):
    accuracy_all = []
    coverage_all = []
    train_all = []
    cor_train_all = []
    for seed in range(1, num_runs + 1):
        acc, cov, tp, ctp = semi_supervised_per_class(data_type, distance_type, metric, starting_sample_size, high_confidence_percentile, low_confidence_percentile, batch_size=batch_size, seed=seed)
        accuracy_all.append(acc)
        coverage_all.append(cov)
        train_all.append(tp)
        cor_train_all.append(ctp)
    class_keys = sorted(accuracy_all[0].keys())
    accuracy_avg = {}
    coverage_avg = {}
    for k in class_keys:
        accuracy_k = np.array([acc[k] for acc in accuracy_all])
        coverage_k = np.array([cov[k] for cov in coverage_all])
        accuracy_avg[k] = np.mean(accuracy_k, axis=0)
        coverage_avg[k] = np.mean(coverage_k, axis=0)
    train_all = np.array(train_all)
    cor_train_all = np.array(cor_train_all)
    train_avg = np.round(np.mean(train_all, axis=0)).astype(int)
    cor_train_avg = np.round(np.mean(cor_train_all, axis=0)).astype(int)
    return coverage_avg, accuracy_avg, train_avg, cor_train_avg

def plot_all_classes_by_batch(coverage_results, accuracy_results):
    fig, ax1 = plt.subplots(figsize=(6, 4))

    ax2 = ax1.twinx()
    ax1.plot(accuracy_results, label='Accuracy', color='blue')
    ax2.plot(coverage_results, label='Coverage', color='green', linestyle='--')

    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Accuracy (%)', color='blue')
    ax2.set_ylabel('Coverage (%)', color='green')

    ax1.tick_params(axis='y', colors='blue')
    ax2.tick_params(axis='y', colors='green')

    ax1.set_ylim(0, 100)
    ax2.set_ylim(0, 100)

    plt.show()

def plot_per_class_by_batch(coverage_results_per_class, accuracy_results_per_class):
    unique_classes = sorted([label for label in accuracy_results_per_class.keys() if label != -1])
    color_map = {label: plt.cm.tab20(i / 20) for i, label in enumerate(range(20))}
    colors = [color_map[label] for label in unique_classes]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()

    for i, class_label in enumerate(unique_classes):
        color = colors[i % len(colors)]
        ax1.plot(accuracy_results_per_class[class_label], label=f'Acc Class {class_label}', color=color)
        ax2.plot(coverage_results_per_class[class_label], label=f'Cov Class {class_label}', color=color, linestyle='--')

    if -1 in accuracy_results_per_class:
        ax1.plot(accuracy_results_per_class[-1], label='Acc Total', color='black', linewidth=2)
        ax2.plot(coverage_results_per_class[-1], label='Cov Total', color='black', linestyle='--', linewidth=2)

    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Accuracy (%) —')
    ax2.set_ylabel('Coverage (%) --')

    ax1.tick_params(axis='y')
    ax2.tick_params(axis='y')

    ax1.set_ylim(0, 100)
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    plt.show()