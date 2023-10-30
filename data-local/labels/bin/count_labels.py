import os
import pandas as pd
import numpy as np


def count_label_file(path_to_file):
    df = pd.read_csv(path_to_file, sep=' ', header=None)

    for client in range(50):
        x = df[(df.iloc[:, 3] == 1) * (df.iloc[:, 2] == client)]
        print(len(x))
        uniques = np.unique(x.values[:, 1], return_counts=True)
        print(len(uniques[1]))


def main():
    dataset = 'miniimagenet'
    filename = '5000_labels_50_clients_50_labelledclients_iid_0_split_0.txt'

    cwd = os.getcwd()
    labels_dir = cwd.rstrip('/bin')
    path = os.path.join(dataset, 'train', filename)

    count_label_file(os.path.join(labels_dir, path))


if __name__ == '__main__':
    main()
