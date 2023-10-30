import numpy as np
import os
import pandas as pd


def check_label_file(path_to_file):
    df = pd.read_csv(path_to_file, sep=' ', header=None)



    for client in range(100):
        client_data_u = df[(df.iloc[:, 3] == 0) * (df.iloc[:, 2] == client)]
        client_data_l = df[(df.iloc[:, 3] == 1) * (df.iloc[:, 2] == client)]
        print(f'client {client}: {len(client_data_l)} labelled, {len(client_data_u)} unlabelled.')

        class_names = np.unique(client_data_l.iloc[:, 1].values, return_counts=True)
        print(class_names[0])
        print(class_names[1])
        class_names = np.unique(client_data_u.iloc[:, 1].values, return_counts=True)
        print(class_names[1])
        # assert (class_names[1] == 1).all()

    print(client_data_l)




def main():
    dataset = 'cifar10'
    filename = '1000_labels_100_clients_50_labelledclients_noniid_7_split_0.txt'

    cwd = os.getcwd()
    labels_dir = cwd.rstrip('/bin')
    path = os.path.join(dataset, 'train', filename)

    check_label_file(os.path.join(labels_dir, path))
    # ptf = os.path.join(labels_dir, dataset, 'train', 'train_labels.txt')
    # df = pd.read_csv(ptf, sep=' ', header=None)
    # print(len(np.unique(df.iloc[:, 0].values, return_counts=True)[1]))


if __name__ == '__main__':
    main()
