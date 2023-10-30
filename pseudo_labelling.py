import faiss
import numpy as np
from scipy import sparse, stats
import time
import torch
import torch.nn.functional as F


def extract_features(dataloader, model, device, isMT=False):
    model.eval()
    embeddings_all, labels_all = [], []

    with torch.no_grad():
        for X, y, _, _, _ in dataloader:
            if isMT:
                X = X[0]
            X = X.to(device)

            *_, feats = model(X)

            embeddings_all.append(feats.cpu().numpy())
            labels_all.append(y.cpu().numpy())

        embeddings_all = np.vstack(embeddings_all)
        labels_all = np.hstack(labels_all)

    return embeddings_all, labels_all


def compute_plabels(X, labels, labeled_idx, args, num_classes=10):
    print('Updating pseudo-labels...')
    alpha = 0.99

    # kNN search for the graph
    d = X.shape[1]
    if args.index_type == 'FlatIP':
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = int(torch.cuda.device_count()) - 1
        index = faiss.GpuIndexFlatIP(res, d, flat_config)  # build the index

    elif args.index_type == 'LSH':
        print(f'using LSH with {args.n_bits} bits.')
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexLSH(d, args.n_bits))

    else:
        raise ValueError('Incorrect index type.')

    faiss.normalize_L2(X)
    index.add(X)
    N = X.shape[0]

    c = time.time()
    D, I = index.search(X, args.dfs_k + 1)
    elapsed = time.time() - c
    print('kNN Search done in %d seconds' % elapsed)

    # Create the graph
    if args.index_type == 'LSH':
        D = np.cos(np.pi * D / args.n_bits)

    D = D[:, 1:] ** 3
    I = I[:, 1:]
    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx, (args.dfs_k, 1)).T
    W = sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
    W = W + W.T

    # Normalize the graph
    W = W - sparse.diags(W.diagonal())
    S = W.sum(axis=1)
    S[S == 0] = 1
    D = np.array(1. / np.sqrt(S))
    D = sparse.diags(D.reshape(-1))
    Wn = D * W * D

    Y = np.zeros((N, num_classes))
    for i in range(num_classes):
        cur_idx = labeled_idx[np.where(labels[labeled_idx] == i)]
        if cur_idx.shape[0] > 0:
            y = np.zeros((N,))
            y[cur_idx] = 1.0 / cur_idx.shape[0]
            Y[:, i] = np.copy(y)

    Z = np.linalg.inv(np.eye(Wn.shape[0]) - alpha * Wn).dot(Y)

    # Handle numerical errors
    Z[Z < 0] = 0

    probs_l1 = F.normalize(torch.tensor(Z), 1).numpy()
    probs_l1[probs_l1 < 0] = 0
    entropy = stats.entropy(probs_l1.T)
    entropy[entropy != entropy] = np.log(num_classes)
    entropy[np.isinf(entropy)] = np.log(num_classes)
    weights = 1 - entropy / np.log(num_classes)
    weights = weights / np.max(weights)
    p_labels = np.argmax(probs_l1, 1)

    correct_idx = (p_labels == labels)
    acc = correct_idx.mean()

    class_acc = []
    for i in range(num_classes):
        if (labels == i).any():
            class_acc.append((correct_idx * (labels == i)).sum() / (labels == i).sum())
        else:
            class_acc.append(-1)

    p_labels[labeled_idx] = labels[labeled_idx]
    weights[labeled_idx] = 1.0
    p_weights = weights.tolist()

    # Compute the weight for each class
    class_weights = np.ones((num_classes,), dtype=np.float32)

    for i in range(num_classes):
        cur_idx = np.where(np.asarray(p_labels) == i)[0]
        if cur_idx.size > 0:
            class_weights[i] = (float(labels.shape[0]) / num_classes) / cur_idx.size

    p_weight_correct = np.array(p_weights)[correct_idx].mean()
    p_weight_incorrect = np.array(p_weights)[~correct_idx].mean()

    statistics = {'p_label_accuracy': acc, 'p_weight_correct': p_weight_correct, 'p_weight_incorrect': p_weight_incorrect}

    return (p_labels, p_weights, class_weights), statistics


def run_psuedo_labelling(sampled_clients, feature_model, device, args):
    # Extract features and update the pseudolabels
    print('Extracting features...')
    feat_list, labels_list, idxs_list = [], [], []
    increment = 0
    for _, train_loader_noshuff, labelled_idx in sampled_clients:
        client_feats, client_labels = extract_features(train_loader_noshuff, feature_model, device, isMT=args.isMT)
        feat_list.append(client_feats)
        labels_list.append(client_labels)
        idxs_list.append(labelled_idx + increment)
        if args.lp_computation == 'global':
            increment += len(client_feats)

    num_classes = 10 if args.dataset == 'cifar10' else 100
    if args.lp_computation == 'global':
        feats = np.vstack(feat_list)
        labels = np.hstack(labels_list).astype(int)
        idxs = np.hstack(idxs_list).astype(int)
        (p_labels, p_weights, class_weights), statistics = compute_plabels(feats, labels, idxs, args,
                                                                           num_classes=num_classes)
        sel_acc = statistics['p_label_accuracy']
        increment = 0
        for j, (train_loader, _, _) in enumerate(sampled_clients):
            left_lim, right_lim = increment, increment + len(train_loader.dataset)
            print(f'Updating client {j} plables')
            assert (right_lim - left_lim) == len(train_loader.dataset)
            train_loader.dataset.update_plabels(p_labels[left_lim:right_lim], p_weights[left_lim:right_lim], class_weights)
            increment = right_lim
        print(f'Pseudo-label accuracy: {sel_acc:.4f}')

        return statistics

    else:
        for feats, labels, idxs, (train_loader, _, _) in zip(feat_list, labels_list, idxs_list, sampled_clients):

            (p_labels, p_weights, class_weights), statistics = compute_plabels(feats, labels, idxs, args,
                                                                               num_classes=num_classes)
            sel_acc = statistics['p_label_accuracy']

            train_loader.dataset.update_plabels(p_labels, p_weights, class_weights)
            print(f'selection accuracy: {sel_acc:.4f}')

        return statistics


def generate_network_plabels(clients, model, device, args):
    statistics = {'p_label_accuracy': None, 'p_weight_correct': None, 'p_weight_incorrect': None}
    num_classes = 10 if args.dataset == 'cifar10' else 100
    print('Using network to pseudo label...')
    model.eval()
    sel_accs = []
    correct_weights, incorrect_weights = [], []
    for train_loader, train_loader_noshuff, labelled_idxs in clients:
        p_labels, sample_weights, true_labels = [], [], []
        with torch.no_grad():
            for X, y, _, _, _ in train_loader_noshuff:
                X = X.to(device)
                preds, *_ = model(X)
                preds_cpu = preds.cpu().numpy()
                # compute pseudo label
                p_labels.append(preds_cpu.argmax(axis=1))
                # compute sample weight
                preds_normalized = preds_cpu / preds_cpu.sum(axis=1, keepdims=True)
                entropy = stats.entropy(preds_normalized, axis=1)
                entropy[entropy != entropy] = np.log(num_classes)
                entropy[np.isinf(entropy)] = np.log(num_classes)
                weights = 1 - entropy / np.log(num_classes)
                sample_weights.append(weights)
                true_labels.append(y.cpu().numpy())

        p_labels = np.hstack(p_labels)
        sample_weights = np.hstack(sample_weights)
        true_labels = np.hstack(true_labels)

        if len(labelled_idxs) > 0:
            p_labels[labelled_idxs] = true_labels[labelled_idxs]
            sample_weights[labelled_idxs] = 1

        sel_accs.append(np.mean(true_labels == p_labels))
        correct_weights.append(np.mean(sample_weights[true_labels == p_labels]))
        incorrect_weights.append(np.mean(sample_weights[true_labels != p_labels]))

        train_loader.dataset.update_plabels(p_labels, sample_weights, [1] * num_classes)

    statistics['p_label_accuracy'] = np.mean(sel_accs)
    statistics['p_weight_correct'] = np.mean(correct_weights)
    statistics['p_weight_incorrect'] = np.mean(incorrect_weights)
    print(f'Pseudo labelling accuracy: {np.mean(sel_accs):.4f}')
    return statistics