import numpy as np
import os
import random

import pandas as pd


def make_label_dist_matrix(base_vec, num_clients):
    label_dist = []
    for _ in range(num_clients):
        label_dist.append(base_vec.copy())
        last = base_vec.pop(-1)
        base_vec.insert(0, last)
    return np.array(label_dist)


def assign_classes(num_classes_per_client, num_classes, num_clients):
    p_mass = 1 / num_classes_per_client
    label_dist = []
    for _ in range(num_clients // num_classes):
        idxs = random.sample(range(num_classes), num_classes_per_client)
        base_vec = [0] * num_classes
        for i in idxs:
            base_vec[i] = p_mass
        assert sum(base_vec) == 1

        for _ in range(num_classes):
            label_dist.append(base_vec.copy())
            last = base_vec.pop(-1)
            base_vec.insert(0, last)

    return np.array(label_dist)


def make_client_labels(dataset, num_labels, num_clients, num_labelled_clients, iid=True, unlabelled_iid=True, dist_num=None, seed=0):
    # set seed for labelled vs unlabelled split
    random.seed(seed)

    # set number of classes
    if dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'cifar100' or dataset == 'miniimagenet':
        num_classes = 100
    else:
        print('Dataset not recognized.')
        raise ValueError

    # get the labelling distributions for each client
    cwd = os.getcwd()
    train_labels_dir = os.path.join(cwd[:cwd.find('bin')], dataset, 'train')
    if iid:
        iid_str, dist_num = 'iid', 0
        client_label_dist = (1 / num_classes) * np.ones((num_clients, num_classes))
    else:
        assert dist_num is not None
        iid_str = 'noniid'
        label_dist_dir = os.path.join(cwd[:cwd.find('bin')], dataset, 'client_label_distributions')
        dist_filename = f'{dataset}_{num_clients}_{dist_num}.txt'
        client_label_dist = np.loadtxt(os.path.join(label_dist_dir, dist_filename))

    client_label_file = f'{num_labels}_labels_{num_clients}_clients_{num_labelled_clients}_labelledclients_{iid_str}_' \
                        f'{dist_num}_split_{seed}.txt'

    # read all train data, store per class and shuffle
    all_data = {}
    with open(os.path.join(train_labels_dir, 'train_labels.txt'), 'r') as input_file:
        for i, line in enumerate(input_file):
            x, y = line.split(' ')
            try:
                all_data[y.rstrip('\n')].append(x)
            except KeyError:
                all_data[y.rstrip('\n')] = [x]
    for _, xs in all_data.items():
        random.shuffle(xs)

    # each client receives the same number of data points, possibly server receives num_labels data points
    num_train = i + 1
    if num_labelled_clients == 0:
        num_datapoint_per_client = (num_train - num_labels) // num_clients
    else:
        num_datapoint_per_client = num_train // num_clients

    if unlabelled_iid:
        client_unlabelled_counts = num_datapoint_per_client * (1 / num_classes) * np.ones((num_clients, num_classes))
    else:
        client_unlabelled_counts = num_datapoint_per_client * client_label_dist

    print(num_datapoint_per_client)

    clients = list(range(num_clients))
    if num_labelled_clients == 0:
        clients.insert(0, 'server')
    datapoint_assignment = {}
    round_up = [1] * len(clients)
    for i, class_name in enumerate(sorted(all_data.keys())):
        right_lim = 0
        for client in clients:
            if client == 'server':
                to_receive = num_labels // num_classes
                to_label = to_receive
            else:
                # last client gets all
                count = client_unlabelled_counts[client][i]
                to_receive = int(count) + np.random.binomial(1, count - int(count)) + (client == num_clients - 1) * 1000
                if client < num_labelled_clients:
                    to_label = (num_labels // num_labelled_clients) * client_label_dist[client][i]
                    if int(to_label) != to_label:
                        to_label = int(to_label) + round_up[client]
                        round_up[client] = 1 - round_up[client]
                else:
                    to_label = 0

                to_label = int(to_label)

            for x in all_data[class_name][right_lim: right_lim + to_label]:
                datapoint_assignment[x] = client, 1

            for x in all_data[class_name][right_lim + to_label: right_lim + to_receive]:
                datapoint_assignment[x] = client, 0

            right_lim += to_receive

    with open(os.path.join(train_labels_dir, 'train_labels.txt'), 'r') as input_file:
        with open(os.path.join(train_labels_dir, client_label_file), 'w') as output_file:
            for i, line in enumerate(input_file):
                x = line.split(' ')[0]
                client, labelled = datapoint_assignment[x]
                output_file.write(line.rstrip('\n') + f" {client} {labelled}" + '\n')


def main():
    cwd = os.getcwd()

    dataset = 'cifar10'
    num_classes = 10 if dataset == 'cifar10' else 100
    num_labels = 1000
    num_clients = 100
    num_labelled_clients = 100
    iid = True
    if not iid:
        dist_num = 7
        # vec = [0.50,0.15,0.03,0.03,0.03,0.02,0.03,0.03,0.03,0.15]
        # vec = [0.2] * 5 + [0] * 2
        # dist_matrix = make_label_dist_matrix(vec, 100)
        dist_matrix = assign_classes(3, num_classes, num_clients)
        label_dist_dir = os.path.join(cwd[:cwd.find('bin')], dataset, 'client_label_distributions')
        client_label_file = f'{dataset}_{num_clients}_{dist_num}.txt'

        np.savetxt(os.path.join(label_dist_dir, client_label_file), dist_matrix)
    else:
        dist_num = 0

    for split in [3, 4, 5]:
        make_client_labels(dataset, num_labels, num_clients, num_labelled_clients, iid=iid, unlabelled_iid=True, dist_num=dist_num, seed=split)


if __name__ == "__main__":
    # print(assign_classes(5, 10, 20))
    main()
    # cwd = os.getcwd()
    # base_dir = os.path.join(cwd[:cwd.find('bin')], 'cifar10', 'train')
    # filename = '5000_labels_100_clients_100_labelledclients_noniid_1_split_0.txt'
    # df = pd.read_csv(os.path.join(base_dir, filename), header=None, sep=' ')
    # client = 2
    # filt = df[(df.iloc[:, 2] == client) * (df.iloc[:, 3] == 1)]
    # print(len(filt))
    # print(filt)
    # with open('/home/jscott/Documents/torch-federated-LP-DeepSSL/run_settings.txt', 'w') as f:
    #     for num_labelled_clients in [0, 50, 100]:
    #         for num_samples in [1000, 5000]:
    #             for dist_num in [0, 1]:
    #                 for split in [0, 1, 2]:
    #                     setting = f'{num_labelled_clients}_{num_samples}_{dist_num}_{split}'
    #                     f.write(setting + '\n')
