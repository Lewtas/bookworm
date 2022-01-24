import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from NBC import *


def load_data():
    ''' Include dataset iris from sklearn'''
    data = pd.read_csv('book_data.csv', encoding='latin-1')
    data = data.dropna()
    for i in data.index:
        data['label'][i] = data['label'][i].replace('[', '')
        data['label'][i] = data['label'][i].replace(']', '')
        data['label'][i] = data['label'][i].split(' ')
        if('' in data['label'][i]):
            data['label'][i] = [x for x in data['label'][i] if x != '']
        data['label'][i] = [float(x) for x in data['label'][i]]
    book_data = np.array([i for i in data['label']])
    book_name = np.array([[i] for i in data['message']])

    return book_name, book_data


def euclidian_dist(x_known, x_unknown):
    """
    This function calculates euclidian distance between each pairs of known and unknown points
    """
    num_pred = x_unknown.shape[0]
    num_data = x_known.shape[0]

    dists = np.empty((num_pred, num_data))

    for i in range(num_pred):
        for j in range(num_data):
            # calculate euclidian distance here
            dists[i, j] = np.sqrt(np.sum((x_unknown[i]-x_known[j])**2))

    return dists


def k_nearest_labels(dists, y_known, k):
    """
    This function returns labels of k-nearest neighbours to each sample for unknown data.
    """

    num_pred = dists.shape[0]
    n_nearest = []

    for j in range(num_pred):
        dst = dists[j]

        # count k closest points
        t = k
        if(t >= dst.shape[0]):
            t = dst.shape[0]-1
        closest_y = y_known[np.argpartition(dst, t)[:k]]

        n_nearest.append(closest_y)
    return np.asarray(n_nearest)


def toBook(message):
    k = 4
    book_name, book_kord = load_data()
    test_data = toChance(message)
    return k_nearest_labels(euclidian_dist(book_kord, test_data), book_name, k)
