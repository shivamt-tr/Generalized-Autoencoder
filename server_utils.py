# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 20:05:23 2019

@author: tripa
"""

import math
import numpy as np
from FCM import FCM
import tensorflow as tf
from scipy.special import comb
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs, make_circles, make_swiss_roll


def load_data(dataset, use_transpose=False):

    if dataset == 'mnist':
        # Load the data, already shuffled
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        # x_train, x_test = x_train / 255.0, x_test / 255.0
        return x_train.reshape(-1, 784), y_train

    if dataset == 'circle':
        return make_circles(n_samples=500, factor=0.8,
                            shuffle=True, random_state=None)

    if dataset == 'swissroll':
        return make_swiss_roll(n_samples=500, random_state=None)

    if dataset == 'iris':
        iris = load_iris()
        return iris['data'], iris['target']

    if dataset == 'augmented_iris':
        iris = load_iris()
        data = iris['data']

        # Add noise to the data
        for i in range(2):
            # Add random noise with mean=0 and variance=0.1 (i.e. std=0.01)
            noise = np.random.normal(loc=0, scale=0.1, size=data.shape[0])
            data = np.c_[data, noise]

        return data, iris['target']

    if dataset == 'blobs':
        return make_blobs(n_samples=1000, n_features=6,
                          centers=2, cluster_std=1.0,
                          shuffle=True, random_state=None)
    
    if dataset == 'synthetic' or dataset == 'synthetic2':
        
        centers = [(2, 2, 2), (5, 2, 4), (5, 2, 8)]
        data, labels = make_blobs(n_samples=500, n_features=3,
                                  centers=centers, cluster_std=1.0,
                                  shuffle=True, random_state=None)
        data = np.vstack((data, [5, 2, 12], [5, 8, 10]))
        labels = np.append(labels, [3, 4])
        
        if dataset == 'synthetic':
            return data, labels
        else:
            # Add noise to the data
            for i in range(2):
                # Add random noise with mean=0 and variance=0.1 (i.e. std=0.01)
                noise = np.random.normal(loc=0, scale=0.1, size=data.shape[0])
                data = np.c_[data, noise]
            return data, labels
        
    return csv_to_numpy(dataset + '.txt', use_transpose)


def entropy_(Y=np.array([])):

    ent = 0
    _, counts = np.unique(Y, return_counts=True)
    for i in range(len(counts)):
        if counts[i] == 0:
            ent += 0
        else:
            p = counts[i] / len(Y)
            ent += - p * math.log(p, 2)
    return ent


def cross_entropy(Y=np.array([]), C=np.array([])):

    c_e = 0
    u_labels_y, counts_Y = np.unique(Y, return_counts=True)
    u_labels_c, counts_C = np.unique(C, return_counts=True)
    for i in range(len(counts_C)):
        p = counts_C[i] / len(C)
        ent = 0
        for j in range(len(counts_Y)):
            a = np.where(C == u_labels_c[i])
            b = np.where(Y == u_labels_y[j])
            if len(np.intersect1d(a, b)) == 0:
                ent += 0
            else:
                p_y = len(np.intersect1d(a, b)) / len(a[0])
                ent += p_y * math.log(p_y, 2)

        c_e += - p * ent
    return c_e


def normalized_mutual_information(Y=np.array([]), C=np.array([])):

    return 2*(entropy_(Y) - cross_entropy(Y, C)) / (entropy_(Y) + entropy_(C))


def normalize(X=np.array([]), use_zscore=False):

    if use_zscore:
        # Apply z_score along each column (accomplished by using axis=0)
        return np.apply_along_axis(lambda x: (x - np.mean(x)) /
                                   np.std(x), 0, X)
    else:
        # Apply 0-1 normalization along each column
        # (accomplished by using axis=0)
        return np.apply_along_axis(lambda x: (x - min(x)) / (max(x) - min(x))
                                   if max(x) != min(x)
                                   else (x - min(x)) / max(x) if max(x) != 0
                                   else 1, 0, X)


def csv_to_numpy(filename, use_transpose):

    f = open(filename, "r").readlines()
    data_list = list()
    labels_list = list()

    if not use_transpose:

        for x in f:
            # Strip and split on ',' and take all but last feature as data
            data_list.append(list(map(float, x.strip().split(',')[:-1])))
            # Strip and split on ',' and take last feature as labels
            labels_list.append(x.strip().split(',')[-1])

    else:

        # Read column wise
        rows = [[x for x in line.split(',')] for line in f]
        cols = [list(col) for col in zip(*rows)]

        for x in cols:
            # Strip and split on ',' and take all but first element as data
            data_list.append(list(map(float, x[1:])))
            # Strip and split on ',' and take first element as labels
            labels_list.append(x[0])

    return np.array(data_list), np.array(labels_list)


def missclassification_error(X=np.array([]), y=np.array([]), n_splits=10):

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    sc = np.empty((n_splits))
    # Enumerate splits, do enumeration instead of 'i=0'
    for i, (train, test) in enumerate(kfold.split(X)):
        neigh = KNeighborsClassifier(n_neighbors=1)
        neigh.fit(X[train], y[train])
        y_pred = neigh.predict(X[test])
        sc[i] = accuracy_score(y[test], y_pred)

    return 100 - 100 * np.mean(sc)


def rand_index(data, data_capped, som_1, som_2):

    # All unordered unique pairs
    combinations = [(i, j)
                    for i in range(data.shape[0])
                    for j in range(i+1, data.shape[0])]

    a, b, c, d = 0, 0, 0, 0
    for combin in combinations:
        q1_1 = som_1.find_bmu(data[combin[0]])
        q1_2 = som_2.find_bmu(data_capped[combin[0]])
        q2_1 = som_1.find_bmu(data[combin[1]])
        q2_2 = som_2.find_bmu(data_capped[combin[1]])

        if(q1_1 == q2_1 and q1_2 == q2_2):
            a += 1
            continue
        if(q1_1 != q2_1 and q1_2 != q2_2):
            b += 1
            continue
        if(q1_1 == q2_1 and q1_2 != q2_2):
            c += 1
            continue
        if(q1_1 != q2_1 and q1_2 == q2_2):
            d += 1
            continue

    return (a+b) / (a+b+c+d)


def _comb2(n):
    # the exact version is faster for k == 2: use it by default globally in
    # this module instead of the float approximate variant
    return comb(n, 2, exact=1)


def pos_1d(x, y, map_size):
    '''
    Given the position (x,y), this function returns the
    position of 2D SOM as node number, with value 0
    for first node, 1 for second node, and so on.
    '''
    return x*map_size[0] + y


def adjusted_rand_index(X, Y):

    contingency = np.zeros(shape=(len(X), len(Y)))

    for i in range(len(X)):
        contingency[int(X[i])][int(Y[i])] += 1

    sum_comb_c = sum(_comb2(n_c) for n_c in np.ravel(contingency.sum(axis=1)))
    sum_comb_k = sum(_comb2(n_k) for n_k in np.ravel(contingency.sum(axis=0)))

    sum_comb = 0
    for i in range(contingency.shape[0]):
        for j in range(contingency.shape[1]):
            sum_comb += _comb2(contingency[i, j])

    # sum_comb = sum(_comb2(n_ij) for n_ij in contingency.data)

    prod_comb = (sum_comb_c * sum_comb_k) / _comb2(len(X))
    mean_comb = (sum_comb_k + sum_comb_c) / 2.

    return (sum_comb - prod_comb) / (mean_comb - prod_comb)


def adjusted_rand_index_som(data, data_capped, som_1, som_2):

    # Create a zero initialized contingency matrix with
    # shape = [n_nodes in SOM_1, n_nodes in SOM_2]
    contingency = np.zeros(shape=(
            som_1.map_size[0] * som_1.map_size[1],
            som_2.map_size[0] * som_2.map_size[1]
            ))

    for i in range(data.shape[0]):
        q1 = som_1.find_bmu(data[i])
        q2 = som_2.find_bmu(data_capped[i])

        contingency[pos_1d(*q1, som_1.map_size),
                    pos_1d(*q2, som_2.map_size)] += 1

    sum_comb_c = sum(_comb2(n_c) for n_c in np.ravel(contingency.sum(axis=1)))
    sum_comb_k = sum(_comb2(n_k) for n_k in np.ravel(contingency.sum(axis=0)))

    sum_comb = 0
    for i in range(contingency.shape[0]):
        for j in range(contingency.shape[1]):
            sum_comb += _comb2(contingency[i, j])

    # sum_comb = sum(_comb2(n_ij) for n_ij in contingency.data)

    prod_comb = (sum_comb_c * sum_comb_k) / _comb2(data.shape[0])
    mean_comb = (sum_comb_k + sum_comb_c) / 2.

    return (sum_comb - prod_comb) / (mean_comb - prod_comb), contingency


def adjusted_rand_index_minisom(data, data_capped, som_1, som_2):

    # Create a zero initialized contingency matrix with
    # shape = [n_nodes in SOM_1, n_nodes in SOM_2]
    contingency = np.zeros(shape=(81, 81))

    for i in range(data.shape[0]):
        q1 = som_1.winner(data[i])
        q2 = som_2.winner(data_capped[i])

        contingency[pos_1d(*q1, (9, 9)),
                    pos_1d(*q2, (9, 9))] += 1

    sum_comb_c = sum(_comb2(n_c) for n_c in np.ravel(contingency.sum(axis=1)))
    sum_comb_k = sum(_comb2(n_k) for n_k in np.ravel(contingency.sum(axis=0)))

    sum_comb = 0
    for i in range(contingency.shape[0]):
        for j in range(contingency.shape[1]):
            sum_comb += _comb2(contingency[i, j])

    # sum_comb = sum(_comb2(n_ij) for n_ij in contingency.data)

    prod_comb = (sum_comb_c * sum_comb_k) / _comb2(data.shape[0])
    mean_comb = (sum_comb_k + sum_comb_c) / 2.

    return (sum_comb - prod_comb) / (mean_comb - prod_comb), contingency


def sammons_error(data, data_capped):

    se = 0
    d_star_sum = 0
    for i in range(len(data)):
        for j in range(i+1, len(data)):
            d_ = np.linalg.norm(data_capped[i] - data_capped[j])
            d_star = np.linalg.norm(data[i] - data[j])
            d_star_sum += d_star
            # Handle divide by 0 exception, skip the iteration
            if d_star == 0:
                continue
            se += (d_ - d_star) ** 2 / d_star

    return se / d_star_sum


def cluster_preserving_index(data, data_capped, n_cluster):

    fcm_original = FCM(data, c=n_cluster, m=2, max_iter=1000).fit()

    # Partion matrix for original data is used to initalize
    # partition matrix for reduced data
    fcm_reduced = FCM(data_capped, c=n_cluster, m=2, max_iter=1000)
    fcm_reduced.memberships = fcm_original.memberships

    # Run FCM on reduced data
    fcm_reduced.fit()

    # Confusion matrix
    cm = np.zeros((n_cluster, n_cluster))
    for i in range(len(data)):
        l_ = int(fcm_original.cluster_labels[i])
        m_ = int(fcm_reduced.cluster_labels[i])
        cm[l_][m_] += 1

    # Normalize confusion matrix by dividing each element by the sum of its row
    cm_normalized = cm.copy()
    cm_normalized /= cm_normalized.sum(axis=1)[:, None]

    # Create a similar matrix
    realigned_cm = np.empty_like(cm)

    for i in range(n_cluster):

        # Copy the 'maxcol'th column (computed by cm_normalized) of cm
        # into the 'i'th column of realigned_cm
        maxcol = np.argmax(cm_normalized[i])
        realigned_cm[:, i] = cm[:, maxcol]

        # Replace all entries in the maxcol column of cm_normalized by -1
        cm_normalized[:, maxcol] = -1

    off_diagonal_sum = np.sum(realigned_cm) - np.trace(realigned_cm)

    return 100 - 100 * (off_diagonal_sum / len(data))


def similarity(gd1, gd2):

    diss = 0
    for i in range(gd1.shape[0]):
        for j in range(gd1.shape[1]):
            diss += abs(gd1[i][j] - gd2[i][j]) / (gd1[i][j] + gd2[i][j])

    return 1 - (diss / (gd1.shape[0] * gd1.shape[1]))


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA.")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y, l


# x = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
# y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3])
