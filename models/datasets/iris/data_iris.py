import os
import csv
import numpy as np


def get_iris_data():
    data = []
    dir_path = os.path.dirname(__file__)
    file_path = os.path.join(dir_path, 'Iris.csv')
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            data.append(row)
        data.pop(0)
    features = []
    labels = []
    for d in data:
        t_f = [float(i) for i in d[1:5]]
        if d[-1] == 'Iris-setosa':
            t_l = [0, 0, 1]
        elif d[-1] == 'Iris-versicolor':
            t_l = [0, 1, 0]
        else:
            t_l = [1, 0, 0]
        features.append(t_f)
        labels.append(t_l)
        # the below lines randomize the input
        features.reverse()
        labels.reverse()
    return {'f': np.array(np.asarray(features)).reshape(len(features), 4),
            'l': np.array(np.asarray(labels)).reshape(len(labels), 3)}


def get_test_data():
    tdata = get_iris_data()
    test_features = []
    test_labels = []
    for i in np.random.randint(0, 150, size=30):
        test_features.append(tdata['f'][i])
        test_labels.append(tdata['l'][i])
    return {'f': np.array(np.asarray(test_features)).reshape(len(test_features), 4),
            'l': np.array(np.asarray(test_labels)).reshape(len(test_labels), 3)}


def create_batches(ar, bsize):
    # Courtesy of http://stackoverflow.com/a/8290508
    l = len(ar)
    batches = []
    for ndx in range(0, l, bsize):
        batches.append(ar[ndx:min(ndx+bsize, l)])
    return batches

if __name__ == '__main__':
    print(get_iris_data())
    print(get_test_data())