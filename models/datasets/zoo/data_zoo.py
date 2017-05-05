import os
import csv
import numpy as np


def get_zoo_data():
    data = []
    dir_path = os.path.dirname(__file__)
    file_path = os.path.join(dir_path, 'zoo.csv')
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            data.append(row)
        data.pop(0)
    features = []
    labels = []
    zoo_class = {
        '1': [0, 0, 0, 0, 0, 0, 1],
        '2': [0, 0, 0, 0, 0, 1, 0],
        '3': [0, 0, 0, 0, 1, 0, 0],
        '4': [0, 0, 0, 1, 0, 0, 0],
        '5': [0, 0, 1, 0, 0, 0, 0],
        '6': [0, 1, 0, 0, 0, 0, 0],
        '7': [1, 0, 0, 0, 0, 0, 0],
    }
    for d in data:
        t_f = [float(i) for i in d[1:17]]
        t_l = zoo_class[d[-1]]

        features.append(t_f)
        labels.append(t_l)
        # the below lines randomize the input
        features.reverse()
        labels.reverse()
    return {'f': np.array(np.asarray(features)).reshape(len(features), 16),
            'l': np.array(np.asarray(labels)).reshape(len(labels), 7)}


def get_test_data():
    tdata = get_zoo_data()
    test_features = []
    test_labels = []
    for i in np.random.randint(0, 100, size=30):
        test_features.append(tdata['f'][i])
        test_labels.append(tdata['l'][i])
    return {'f': np.array(np.asarray(test_features)).reshape(len(test_features), 16),
            'l': np.array(np.asarray(test_labels)).reshape(len(test_labels), 7)}


def create_batches(ar, bsize):
    # Courtesy of http://stackoverflow.com/a/8290508
    l = len(ar)
    batches = []
    for ndx in range(0, l, bsize):
        batches.append(ar[ndx:min(ndx+bsize, l)])
    return batches

if __name__ == '__main__':
    print(get_zoo_data())
    print(get_test_data())