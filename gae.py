# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 16:03:59 2019

@author: tripa
"""

import sys
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
from FCM import FCM
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import graph_shortest_path
from server_utils import (load_data, normalize)

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

import seaborn as sns
sns.set_style('ticks')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1,
                rc={"lines.linewidth": 2.5})


def scatter(x, colors, n_classes,
            cluster_centers=None,
            plot_cluster_centers=False):

    palette = np.array(sns.color_palette("hls", n_classes))

    # Create a scatter plot
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(aspect='equal')
    ax.scatter(x[:, 0], x[:, 1], lw=0, s=50,
               c=palette[colors.astype(np.int)])

    # Plot cluster_centers in black
    if plot_cluster_centers:
        ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
                   s=40, c='black')

    ax.axis('tight')

    # We add the labels for each category
    for i in range(n_classes):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=16)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])

    return fig


def scatter_swissroll(x, color):
    
    # Create a scatter plot
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(aspect='equal')

    ax.scatter(x[:, 0], x[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.axis('tight')
    plt.xticks([]), plt.yticks([])
    
    return fig


def scatter_coil(x):

    # Create a scatter plot
    fig = plt.figure(figsize=(6, 6))
    # ax = plt.subplot(aspect='equal')

    for i, point in enumerate(x):
        txt = plt.text(point[0], point[1], str(i+1), fontsize=12)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])

    return fig

# %%


class GeneralizedAutoEncoder(object):

    """Generalized Autoencoder
    Dimensionality reduction by training the latent space.

    Parameters and Attributes
    -------------------------
    data: numpy array
        input data samples
    labels: numpy array
        class labels of input data samples
    layers: list
        list of nodes in each layer
    file_name: string
        save the model with name as file name
    learning_rate: float
        learning_rate to be used for training
    epoch: integer
        maximum number iterations for training
    initializer: ['rn'|'he'|'xavier']
        initializer for initializing the weights and biases
    loss_function: ['e1'|'e2'|'e3'|'e4'|'e5'|'e6'|'e7'|'e8'|'e9']
        loss_function to be used
    display_step: integer
        store the result after these many steps
    use_knn: boolean
        whether to use neighbors
    knn: object
        if use_knn is true, then the knn object
    k: integer
        number of neighbors to be considered in loss function
        for each data sample
    t: float
        t = 2 * sigma * sigma
        sigma is the tuning parameter in the reconstruction weights
    lmbda, alpha, lmbda1, lmbda2: float
        parameters in loss_function
    n_cluster: integer
        number of clusters for kmeans or fcm
    use_kmeans: boolean
        whether to use kmeans on the input data
    kmeans: object
        if use_kmeans is true, then the kmeans object
    use_fcm: boolean
        whether to use the fcm on the input data
    fcm: object
        if use_fcm is true, then the fcm object
    use_geodesic: boolean
        whether to use the geodesic distance as the reconstruction weight
    update_weights_at_step: integer
        if required, this parameter specifies after how many steps
        to update the reconstruction weights based on latent space data
    use_pretrained: boolean
        whether to use pretrained parameters
    file_name: string
        file_name for saving files in the file_name directory

    Attributes
    ----------
    n_layers: integer
        number of layers in the autoencoder
    weights: dict
        dictionary of tensors to hold the weight matrix for each later
    biases: dict
        dictionary of tensors to hold the biases for each layer
    encoder_op: tensor
        tensor to hold the encoder output
    decoder_op: tensor
        tensor to hold the decoder output
    saver: tensor
        tensor to hold the saver for the model
    X : tensor, shape (n_samples, n_features)
        placeholder tensor for feeding the input dataarray,
        where n_samples in the number of samples
        and n_features is the number of features.

    Examples
    --------
    >>> model = GeneralizedAutoEncoder(data, labels, layers,
                                       display_step=1000,
                                       use_knn=True,
                                       knn=knn,
                                       k=i, t=j,
                                       alpha=0.1, lmbda=0.1,
                                       n_cluster=n_cluster,
                                       use_kmeans=True,
                                       kmeans=kmeans,
                                       loss_function='e8',
                                       file_name=file_name)
    >>> model.train(--------)
    >>> model.test(---------)

    References(Edit This and add more)
    ----------
    .. [1] Wang, W.; Huang, Y.; Wang, Y.; and Wang, L. 2014.Generalized
        autoencoder: A neural network framework for dimensionality reduction
    """
    def __init__(self, data=np.array([]),
                 labels=np.array([]),
                 layers=[],
                 initializer="rn",
                 loss_function='e1',
                 display_step=1000,
                 use_knn=False,
                 knn=None,
                 k=None, t=None,
                 lmbda=None,
                 alpha=None,
                 lmbda1=None,
                 lmbda2=None,
                 n_cluster=None,
                 use_kmeans=False,
                 kmeans=None,
                 use_fcm=False,
                 fcm=None,
                 use_geodesic=False,
                 geodesic_distance=None,
                 update_weights_at_step=None,
                 use_pretrained=False,
                 file_name="default_model/"):

        # Reset Graph
        tf.reset_default_graph()
        # np.random.seed(42)
        assert len(layers) > 1
        # Add the dimension of input data as the size of first layer
        layers.insert(0, data.shape[1])
        self.layers = layers
        self.n_layers = len(layers)
        self.file_name = file_name
        self.model_path = "./model/" + self.file_name
        self.graph_path = "./graph/" + self.file_name
        self.data = data
        # Convert labels of all types to unique numbers
        unique_labels, labels_ = np.unique(labels, return_inverse=True)
        self.labels = labels_
        self.n_classes = len(unique_labels)  # Number of classes in data
        self.loss_function = loss_function
        self.display_step = display_step
        if use_knn:
            self.knn = knn
        self.k = k
        self.t = t
        self.alpha = alpha
        self.lmbda = lmbda
        self.lmbda1 = lmbda1
        self.lmbda2 = lmbda2
        self.n_cluster = n_cluster
        self.weights = dict()
        self.biases = dict()
        # To store weights and biases in order to initialize
        # network by pretrained parameters
        self.trained_weights = dict()
        self.trained_biases = dict()
        self.encoder_op = None
        self.decoder_op = None
        self.saver = None
        if(initializer == "rn"):
            self.initializer = tf.random_normal_initializer()
        if(initializer == "he"):
            self.initializer = tf.contrib.layers.variance_scaling_initializer()
        if(initializer == "xavier"):
            self.initializer = tf.contrib.layers.xavier_initializer()
        self.use_kmeans = use_kmeans  # For clustering based loss function
        self.use_fcm = use_fcm  # For clustering based loss function
        if(use_kmeans):
            self.kmeans = kmeans
        if(use_fcm):
            self.fcm = fcm
        self.use_geodesic = use_geodesic
        if self.use_geodesic:
            self.gd = geodesic_distance
        self.update_weights_at_step = update_weights_at_step
        self.use_pretrained = use_pretrained

        # tf Graph input
        self.X = tf.placeholder(tf.float64, shape=(None, self.layers[0]),
                                name="X")

    # Write Documentation
    def initialize_layers(self):
        '''
        To initialize weights and biases dictionary based on 'layers' list.
        The 'layers' list defines the architecture of encoder.
        The decoder is built by traversing the list in reverse.
        Eg. layers = [784, 500, 300, 100, 2]
        For weights dictionary keys are -
        'encoder_1', 'encoder_2', 'encoder_3', 'hidden',
        'decoder_3', 'decoder_2,' 'decoder_1', 'output'
        For biases dictionary keys are -
        'encoder_b1', 'encoder_b2', 'encoder_b3', 'hidden_b',
        'decoder_b3', 'decoder_b2,' 'decoder_b1', 'output_b'
        '''

        # Iterate i = 0, 1, 2 for n_layers = 5
        for i in range(self.n_layers-2):
            self.weights['encoder_'+str(i+1)] = tf.Variable(self.initializer([
                    self.layers[i], self.layers[i+1]
                    ], dtype=tf.float64))
            self.biases['encoder_b'+str(i+1)] = tf.Variable(self.initializer([
                    self.layers[i+1]
                    ], dtype=tf.float64))

        # Initialize hidden layer weights and biases
        self.weights['hidden'] = tf.Variable(self.initializer([
                self.layers[-2], self.layers[-1]
                ], dtype=tf.float64))
        self.biases['hidden_b'] = tf.Variable(self.initializer([
                self.layers[-1]
                ], dtype=tf.float64))

        # Iterate i = 4, 3, 2 for n_layers = 5
        for i in range(self.n_layers-1, 1, -1):
            self.weights['decoder_'+str(i-1)] = tf.Variable(self.initializer([
                    self.layers[i], self.layers[i-1]
                    ], dtype=tf.float64))
            self.biases['decoder_b'+str(i-1)] = tf.Variable(self.initializer([
                    self.layers[i-1]
                    ], dtype=tf.float64))

        # Initialize output layer weights and biases
        self.weights['output'] = tf.Variable(self.initializer([
                self.layers[1], self.layers[0]
                ], dtype=tf.float64))
        self.biases['output_b'] = tf.Variable(self.initializer([
                self.layers[0]
                ], dtype=tf.float64))

    def initialize_by_dict(self):

        # Layer names
        # encoder_1, encoder_2, encoder_3, hidden, decoder_3, decoder_2,
        # decoder_1, output

        # Iterate 0, 1, 2 for n_layers = 5
        for i in range(self.n_layers-2):
            self.weights['encoder_'+str(i+1)] = tf.Variable(
                    self.trained_weights['encoder_'+str(i+1)]
                    )
            self.biases['encoder_b'+str(i+1)] = tf.Variable(
                    self.trained_biases['encoder_b'+str(i+1)]
                    )

        self.weights['hidden'] = tf.Variable(
                self.trained_weights['hidden']
                )
        self.biases['hidden_b'] = tf.Variable(
                self.trained_biases['hidden_b']
                )

        # Iterate 4, 3, 2 for n_layers = 5
        for i in range(self.n_layers-1, 1, -1):
            self.weights['decoder_'+str(i-1)] = tf.Variable(
                    self.trained_weights['decoder_'+str(i-1)]
                    )
            self.biases['decoder_b'+str(i-1)] = tf.Variable(
                    self.trained_biases['decoder_b'+str(i-1)]
                    )

        self.weights['output'] = tf.Variable(
                self.trained_weights['output']
                )
        self.biases['output_b'] = tf.Variable(
                self.trained_biases['output_b']
                )

    # Write Documentation
    def encoder(self, X):
        '''
        Forward pass for encoder
        '''

        # Apply forward pass from input to first layer
        encode = tf.nn.sigmoid(tf.add(tf.matmul(X, self.weights['encoder_1']),
                                      self.biases['encoder_b1']))

        # Forward pass for remaining encoder layers
        # Iterate 1, 2 for n_layers = 5
        for i in range(1, self.n_layers-2):
            next_encode = tf.nn.sigmoid(
                    tf.add(tf.matmul(
                            encode,
                            self.weights['encoder_'+str(i+1)]
                            ), self.biases['encoder_b'+str(i+1)])
                    )

            encode = next_encode

        # Get the latent space output u.e. the encoded result
        hidden = tf.nn.sigmoid(tf.add(tf.matmul(
                encode, self.weights['hidden']
                ), self.biases['hidden_b']))

#        hidden = tf.add(tf.matmul(encode, self.weights['hidden']),
#                        self.biases['hidden_b'])

        return hidden

    def decoder(self, X):
        '''
        Forward pass for decoder
        '''

        # Apply forward pass from hidden layer to first decoder layer
        decode = tf.nn.sigmoid(tf.add(
                tf.matmul(X, self.weights['decoder_'+str(self.n_layers-2)]),
                self.biases['decoder_b'+str(self.n_layers-2)])
            )

        # Forward pass for remaining decoder layers
        # Iterate 3, 2 for n_layers = 5
        for i in range(self.n_layers-2, 1, -1):
            next_decode = tf.nn.sigmoid(tf.add(
                    tf.matmul(decode, self.weights['decoder_'+str(i-1)]),
                    self.biases['decoder_b'+str(i-1)])
            )
            decode = next_decode

        # Get the reconstructed output
        output = tf.nn.sigmoid(tf.add(
                tf.matmul(decode, self.weights['output']),
                self.biases['output_b'])
            )

        return output

    def get_reconstruction_weights_knn(self, N):

        data_ = np.stack(([self.data for _ in range(self.k)]), axis=1)

        # We use axis=2 because the value inside np.sum is
        # of shape (n, k, 784) and we want (n, k)
        return np.exp(-np.sum(np.square(data_ - self.data[N]), axis=2)/self.t)

    def get_reconstruction_weights_cluster(self, cluster_centers):

        data = self.data[:-self.n_cluster]
        n = data.shape[0]
        reconstruction_weights = np.zeros((n, self.n_cluster))

        # Compute reconstruction_weights for each datapoint
        # with respect to 'c' cluster centers
        for i in range(n):
            reconstruction_weights[i, :] = np.exp(-np.sum(np.square(
                    data[i] - cluster_centers), axis=1) / self.t)

        # Normalize along row, divide each row by the sum of
        # reconstruction_weights of that row
        for i in range(self.n_cluster):
            reconstruction_weights[:, i] /= \
                np.sum(reconstruction_weights[:, i])

        return reconstruction_weights

    # Write Documentation
    def get_loss(self, y_true, y_pred):

        # Calculate MSE loss
        if self.loss_function in ['e1', 'e3', 'e7', 'e9',
                                  'e10', 'e10_geodesic']:
            mse_loss = tf.reduce_sum(tf.pow(y_true - y_pred, 2))

        # Get the neighborhood matrix from knn object
        if self.loss_function in ['e2', 'e3', 'e4', 'e5', 'e6']:
            N = self.knn.kneighbors(self.data, return_distance=False)

        # In following cost functions augmented data is received
        # so we split it and calculate the neighborhood matrix
        # from knn object
        if self.loss_function in ['e7', 'e8']:
            N = self.knn.kneighbors(self.data[:-self.n_cluster],
                                    return_distance=False)

        # Calculate the reconstruction weights
        if self.loss_function in ['e2', 'e3', 'e4', 'e6']:
            reconstruction_weights = self.get_reconstruction_weights_knn(N)

        # Calculate the reconstruction weights
        # for clustering based cost functions
        if self.loss_function in ['e8', 'e9']:
            reconstruction_weights = \
                self.get_reconstruction_weights_cluster(
                        self.data[-self.n_cluster:])

        if self.loss_function in ['e2', 'e4', 'e5', 'e6']:

            # Index neighbors using N from input data
            Xj = self.data[N]

            # Repeat reconstruction output 'k' times for each point
            # and stack them by using axis=1 beacause out of 3 dimensions
            # (0,1 and 2), we need to stack along second dimension to make a
            # tensor of shape (n, k, data.shape[1]).
            Xi = tf.stack(([self.decoder_op for i in range(self.k)]), axis=1)

        if self.loss_function in ['e3', 'e4', 'e5', 'e6']:

            # Repeat enocoder output 'k' times for each point
            # and stack them by using axis=1 beacause out of 3 dimensions
            # (0,1 and 2), we need to stack along second dimension to make a
            # tensor of shape (n, k, 2)
            Yi = tf.stack(([self.encoder_op for i in range(self.k)]), axis=1)

            # Index neighbors using N from encoder output
            Yj = tf.gather(self.encoder_op, N)

        if self.loss_function == 'e1':
            return mse_loss

        if self.loss_function == 'e2':

            loss = tf.reduce_sum(tf.multiply(reconstruction_weights,
                                 tf.reduce_sum(tf.pow(Xi - Xj, 2), axis=2)))

            return loss

        if self.loss_function == 'e3':

            loss = tf.reduce_sum(tf.multiply(
                                reconstruction_weights,
                                tf.reduce_sum(tf.pow(Yi - Yj, 2), axis=2)
                                ))

            return mse_loss + self.lmbda*loss

        if self.loss_function == 'e4':

            # Square the difference of Xi and Xj and sum across
            # axis=2, because out of 3 dimensions (0, 1, and 2),
            # we need to sum along the third dimension to make
            # a tensor of shape (n, k).
            # Later, multiply the result by reconstruction_weights and sum.
            loss1 = tf.reduce_sum(tf.multiply(reconstruction_weights,
                                  tf.reduce_sum(tf.pow(Xi - Xj, 2), axis=2)))

            # Square the difference of Yi and Yj and sum across
            # axis=2, because out of 3 dimensions (0, 1, and 2),
            # we need to sum along the third dimension to make
            # a tensor of shape (n, k).
            # Later, multiply the result by reconstruction_weights and sum.
            loss2 = tf.reduce_sum(tf.multiply(reconstruction_weights,
                                  tf.reduce_sum(tf.pow(Yi - Yj, 2), axis=2)))

            loss = loss1 + self.lmbda*loss2

            return loss

        if self.loss_function in ['e5', 'e6']:

            # Calculate norm along axis=2
            loss1 = tf.norm(Xi - Xj, axis=2) - tf.norm(Yi - Yj, axis=2)
            print(loss1.get_shape)

            if self.loss_function == 'e5':
                loss = tf.reduce_sum(tf.pow(loss1, 2))
                return loss

            if self.loss_function == 'e6':
                loss = tf.reduce_sum(tf.pow(tf.multiply(
                                    reconstruction_weights, loss1), 2))
                return loss

        if self.loss_function == 'e7':

            data = self.data[:-self.n_cluster]

            # Calculate reconstruction weights
            data_ = np.stack(([data for _ in range(self.k)]), axis=1)

            # We use axis=2 because the value inside np.sum is
            # of shape (n, k, 784) and we want (n, k)
            reconstruction_weights = np.exp(-np.sum(
                    np.square(data_ - data[N]), axis=2) / self.t)

            # print(reconstruction_weights)

            # Make reconstruction weight lmbda1 times if the neighbor
            # belongs to same cluster, else make it lmbda2 times
            for i in range(data.shape[0]):
                for j in range(self.k):
                    if self.kmeans.labels_[i] == self.kmeans.labels_[j]:
                        reconstruction_weights[i, j] *= self.lmbda1
                    else:
                        reconstruction_weights[i, j] *= -self.lmbda2

            # print(reconstruction_weights)

            # Repeat enocoder output 'k' times for each point
            # and stack them by using axis=1 beacause out of 3 dimensions
            # (0,1 and 2), we need to stack along second dimension to make a
            # tensor of shape (n, k, 2)
            Yi = tf.stack(([self.encoder_op[:-self.n_cluster]
                           for i in range(self.k)]), axis=1)

            # Index neighbors using N from encoder output
            Yj = tf.gather(self.encoder_op[:-self.n_cluster], N)

            loss = mse_loss + tf.reduce_sum(tf.multiply(
                        reconstruction_weights,
                        tf.reduce_sum(tf.pow(Yi - Yj, 2), axis=2)
                        ))

            # Dividing by number of points make it independent of number of
            # points. and makes the choice of learning rate and other
            # parameters easier

            return loss / self.data[:-n_cluster].shape[0]

        if self.loss_function == 'e8':

            # Remove augmented cluster centers and calculate mse loss
            y_true = y_true[:-self.n_cluster]
            y_pred = y_pred[:-self.n_cluster]
            mse_loss = tf.reduce_sum(tf.pow(y_true - y_pred, 2))

            # Index neighbors using N from input data
            Xj = self.data[:-self.n_cluster][N]

            # Repeat reconstruction output 'k' times for each point
            # and stack them by using axis=1 beacause out of 3 dimensions
            # (0,1 and 2), we need to stack along second dimension to make a
            # tensor of shape (n, k, data.shape[1]).
            Xi = tf.stack(([self.decoder_op[:-self.n_cluster]
                            for i in range(self.k)]), axis=1)

            # Repeat enocoder output 'k' times for each point
            # and stack them by using axis=1 beacause out of 3 dimensions
            # (0,1 and 2), we need to stack along second dimension to make a
            # tensor of shape (n, k, 2)
            Yi = tf.stack(([self.encoder_op[:-self.n_cluster]
                            for i in range(self.n_cluster)]), axis=1)

            # Yj stores the encoder output of all cluster centers
            Yj = self.encoder_op[-self.n_cluster:]

            loss1 = (1-self.alpha) * mse_loss + \
                    (self.alpha / self.k) * tf.reduce_sum(tf.pow(Xi - Xj, 2))

            loss2 = tf.reduce_sum(tf.multiply(reconstruction_weights,
                                  tf.reduce_sum(tf.pow(Yi - Yj, 2), axis=2)))

            return loss1 + self.lmbda*loss2

        if self.loss_function == 'e9':

            # Repeat enocoder output 'n_cluster' times for each point
            # and stack them by using axis=1 beacause out of 3 dimensions
            # (0,1 and 2), we need to stack along second dimension to make a
            # tensor of shape (n, n_cluster, 2)
            Yi = tf.stack(([self.encoder_op[:-self.n_cluster]
                            for i in range(self.n_cluster)]), axis=1)

            # Yj stores the encoder output of all cluster centers
            Yj = self.encoder_op[-self.n_cluster:]

            # Reconstruction weights is the membership matrix here
            reconstruction_weights = self.fcm.memberships

            loss = mse_loss + tf.reduce_sum(tf.multiply(
                                    reconstruction_weights,
                                    tf.reduce_sum(tf.pow(Yi - Yj, 2), axis=2)
                                ))

            return loss

        if self.loss_function == 'e10':

            n_points = self.data.shape[0]

            # Calculate D and D_star matrix for sammon's error
            D = self.encoder_op - tf.stack(([self.encoder_op
                                            for i in range(n_points)]), axis=1)
            D_star = self.X - tf.stack(([self.X
                                         for i in range(n_points)]), axis=1)
            D = tf.norm(D, axis=2)
            D_star = tf.norm(D_star, axis=2)

            # Calculate Sammon's error
            sammons_loss = tf.reduce_sum(tf.div_no_nan(tf.pow(D - D_star, 2),
                                                       D_star))
            sammons_loss /= tf.reduce_sum(D_star)  # Divide by D_star Sum
            loss = (mse_loss/n_points) + self.lmbda * sammons_loss

        if self.loss_function == 'e10_geodesic':

            n_points = self.data.shape[0]  # Number of data samples

            # Placeholder for D* (Geodesic distance will be given in session)
            D_star = tf.constant(self.gd, dtype=tf.float64, name='D_star')
            # Construct D by constructing a pair wise matrix
            D = self.encoder_op - tf.stack(([self.encoder_op
                                            for i in range(n_points)]), axis=1)
            # Take norm of each weight vector in tensor D
            # (Shape=nxnxd) to (nxn)
            D = tf.norm(D, axis=2)

            # Calculate Sammon's error as (1/sum(D*)) * (((D-D*)^2) / D*)
            sammons = tf.reduce_sum(tf.div_no_nan(tf.pow(D - D_star, 2),
                                                  D_star))
            sammons /= tf.reduce_sum(D_star)

            loss = (mse_loss/n_points) + self.lmbda * sammons

            return loss

    # Write Documentation
    def train(self, eta=0.1, epochs=100):

        # Create folders to save model and graph
        Path(self.model_path).mkdir(parents=True, exist_ok=True)
        Path(self.graph_path).mkdir(parents=True, exist_ok=True)

        # Initialize layers by dictionary or random
        if self.use_pretrained:
            self.initialize_by_dict()
        else:
            self.initialize_layers()

        # Construct model
        self.encoder_op = self.encoder(self.X)
        self.decoder_op = self.decoder(self.encoder_op)

        # Reconstructed data
        y_pred = self.decoder_op
        # Input data
        y_true = self.X

        # Define loss and optimizer
        loss = self.get_loss(y_true, y_pred)

        optimizer = tf.train.AdamOptimizer(eta, epsilon=0.1).minimize(loss)

        # Create a saver
        self.saver = tf.train.Saver()

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # FileWriter for graph
        tf.summary.FileWriter(self.graph_path, tf.get_default_graph())

        # Start Training
        with tf.Session() as sess:

            # Run the initializer
            sess.run(init)

            # Loss function calculation should not be based on y
            y = np.zeros([len(self.data), 2])

            # Variable to store all the cost
            # adding 1 extra because epoch starts from 1
            cost_ = np.zeros((1+epochs,))

            tic = time.time()

            # Training
            for epoch in range(1, epochs+1):

                # Run optimizer
                y, _, epoch_loss = sess.run([self.encoder_op,
                                             optimizer,
                                             loss],
                                            feed_dict={self.X: self.data})

                cost_[epoch] = epoch_loss

                if epoch % self.display_step == 0 or epoch == 1:
                    print(f'\rEpoch {epoch}/{epochs}',
                          f'Loss {epoch_loss:.4f}', end='')
                    sys.stdout.flush()

                    # Plot cluster_centers for suitable cost functions
                    if self.use_kmeans or self.use_fcm:
                        fig = scatter(y[:-self.n_cluster],
                                      self.labels,
                                      n_classes=self.n_classes,
                                      cluster_centers=y[-self.n_cluster:],
                                      plot_cluster_centers=True)
                    else:
                        # Plot cost
                        plt.figure(figsize=(6, 6))
                        plt.grid(True)
                        # plt.title('Cost')
                        plt.xlabel('Iteration')
                        plt.ylabel('Cost')
                        plt.plot(cost_[1:], linewidth=1, color='blue')
                        plt.savefig('./figures/'+self.file_name+'_cost.png',
                                    format='png', dpi=500)
                        plt.show()
                        plt.close()

                        # fig = scatter(y, self.labels,
                        #               n_classes=self.n_classes)
                        # fig = scatter_coil(y)
                        fig = scatter_swissroll(y, self.labels)

                    # Save figure as png
                    fig.savefig('./figures/'+self.file_name+'_'+str(epoch)+'.png',
                                format='png', dpi=500)
                    plt.show()

#                # Break if the convergence has became slow
#                if epoch > 1 and cost_[epoch-1] - cost_[epoch] < 0.000001:
#
#                    print("Updating eta to ", eta/2)
#                    eta /= 2
#                    if eta < 0.0001:
#                        print(f"Exiting at iteration {epoch}")
#                        break

#                # This part is incomplete
#                if self.update_weights_at_step:
#                    # Update reconstruction weights after steps
#                    # Do not update at epoch 0
#                    if epoch and epoch % self.update_weights_at_step == 0:
#                        # write the logic here
#                        pass
#                        # recon_weights = update_weights()

            print(f'\rTime taken {time.time()-tic:.2f} Seconds')

            if self.use_kmeans or self.use_fcm:
                fig = scatter(y[:-self.n_cluster],
                              self.labels,
                              n_classes=self.n_classes,
                              cluster_centers=y[-self.n_cluster:],
                              plot_cluster_centers=True)
            else:

                # Plot cost
                plt.figure(figsize=(6, 6))
                plt.grid(True)
                # plt.title('Cost')
                plt.xlabel('Iteration')
                plt.ylabel('Cost')
                plt.plot(cost_[1:], linewidth=1, color='blue')
                plt.savefig('./figures/'+self.file_name+'_cost.png',
                            format='png', dpi=500)
                plt.show()
                plt.close()

                # Plot hidden space
                # fig = scatter(y, self.labels, n_classes=self.n_classes)
                # fig = scatter_coil(y)
                fig = scatter_swissroll(y, self.labels)

            # Save figure as png
            fig.savefig('./figures/'+self.file_name+'_result.png',
                        format='png', dpi=500)
            plt.show()

            # Save model (takes time, so don't do it at each iteration)
            self.saver.save(sess, self.model_path+'/model.ckpt')

            # Plot loss
            # Because epoch starts from 1, cost_[0] is not defined
            np.savetxt(self.file_name+'_cost.txt', cost_, delimiter=',')

        return self

    # Testing
    def test(self, dataset='mnist'):

        if self.loss_function in ['e7', 'e8', 'e9']:
            # Here, augmented data is received so we split it
            self.data = self.data[:-self.n_cluster]

        if dataset == 'mnist':
            # Select a random datapoint corresponding to each class label
            data_index = np.array(range(0, len(self.data)))
            unique_labels = np.unique(self.labels)
            test_data = np.empty((len(unique_labels), self.layers[0]))

            for i, n in enumerate(unique_labels):
                test_data[i] = self.data[
                        np.random.choice(data_index[self.labels == n])]

            # Encode and decode images from test set
            # and visualize reconstructions
            # Create a canvas to store plots for original
            # and reconstructed images
            canvas = np.empty((28 * len(unique_labels), 28*2))

            with tf.Session() as sess:

                # Restore model from saver object
                self.saver.restore(sess, self.model_path+'/model.ckpt')

                recon = sess.run(self.decoder_op,
                                 feed_dict={self.X: test_data}
                                 )

                # Display original and reconstructed images
                for i in range(len(unique_labels)):
                    canvas[i*28: (i+1)*28, 0:28] = \
                        test_data[i].reshape([28, 28])
                    canvas[i*28: (i+1)*28, 28:56] = \
                        recon[i].reshape([28, 28])

                # In future, display horizontally instead of vertically
                # print("Canvas with Original and Reconstructed Images")
                plt.figure(figsize=(6, 6))
                plt.axis('off')
                plt.xticks([])
                plt.yticks([])
                plt.imshow(canvas, origin="upper", cmap="gray")
                # Save figure
                plt.savefig('./figures/'+self.file_name+'_reconstruction.png',
                            format='png', dpi=500)
                plt.show()
                plt.close()

        if dataset == 'circle':

            with tf.Session() as sess:

                # Restore model from saver object
                self.saver.restore(sess, self.model_path+'/model.ckpt')

                recon = sess.run(self.decoder_op,
                                 feed_dict={self.X: self.data}
                                 )

                scatter(recon, self.labels, n_classes=self.n_classes)
                plt.show()


# %%

# Load data
dataset = 'wine'
if dataset in ['iris', 'augmented_iris', 'synthetic', 'mnist']:
    data, labels = load_data(dataset)
else:
    data, labels = load_data('./datasets/'+dataset)

data = normalize(data)
# data = data[:100, :]
# labels = labels[:100]

# %%

# Load coil data
f = open('./datasets/coil_1024_object1_zero_one.txt', "r").readlines()
data_list = list()
labels_list = list()

for x in f:
    # Strip and split on ',' and take all but last feature as data
    data_list.append(list(map(float, x.strip().split(',')[:])))
    # Strip and split on ',' and take last feature as labels
    labels_list.append(0)

data, labels = np.array(data_list), np.array(labels_list)

# plt.imshow(data[0].reshape([32, 32]))1


# %%


def floyd_warshall(am):
    n = am.shape[0]
    for k in range(n):
        am = np.minimum(am, am[np.newaxis, k, :] + am[:, k, np.newaxis])

    return am


def geodesic(data, k=5):

    n = len(data)  # Number of samples

    # Get the neighborhood matrix for given k
    knn = NearestNeighbors(n_neighbors=k).fit(data)
    N = knn.kneighbors(data, return_distance=False)

    # Find non-neighbor indices
    not_a_neighbor = np.ones((n, n), dtype=bool)
    for i in range(n):
        for j in N[i]:  # Iterate over all neighbors of data point 'i'
            not_a_neighbor[i, j] = 0  # All these points are neighrbors

    # Build adjacency matrix
    adj_mat = np.stack([data for i in range(len(data))], axis=1) - data
    adj_mat = np.linalg.norm(adj_mat, axis=2)
    adj_mat[not_a_neighbor] = np.inf  # Make non-neighbor pair distance inf
    adj_mat[range(n), range(n)] = 0  # Make diagonal values zero

    return adj_mat, floyd_warshall(adj_mat)


from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_swiss_roll

k = 10  # k is 1 % of number of samples
data, labels = make_swiss_roll(n_samples=1000, random_state=None)
# data = normalize(data)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap=plt.cm.Spectral)
plt.show()

# Calculate geodesic
am, geo = geodesic(data, k)

# %%

layers = [50, 20, 10, 2]

################# Geodesic ########################

file_name = "e10_geodesic"
model = GeneralizedAutoEncoder(data, labels, layers,
                               display_step=100,
                               lmbda=0.1,
                               loss_function='e10_geodesic',
                               use_geodesic=True,
                               geodesic_distance=geo,
                               file_name=file_name).train(eta=1.0,
                                                          epochs=5000)

# %%

layers = [600, 300, 100, 50, 2]

################# Sammon's Cost Function ########################

file_name = "e10"
model = GeneralizedAutoEncoder(data, labels, layers,
                               display_step=1000,
                               lmbda=0,
                               loss_function='e10',
                               file_name=file_name).train(eta=0.001,
                                                          epochs=10000)
# model.test()

# %%

k, t = 10, 50
n_cluster = 10
kmeans = KMeans(n_cluster).fit(data)

data = np.concatenate((data, kmeans.cluster_centers_))

# Compute K Nearest Neighbors, exclude appended cluster centers
knn = NearestNeighbors(n_neighbors=k).fit(data[:-n_cluster])

file_name = "e7_" + str(k) + "_" + str(t)
model = GeneralizedAutoEncoder(data, labels, layers,
                               display_step=1000,
                               use_knn=True,
                               knn=knn,
                               k=k, t=t,
                               lmbda1=0.1, lmbda2=0,
                               n_cluster=n_cluster,
                               use_kmeans=True,
                               kmeans=kmeans,
                               loss_function='e7',
                               file_name=file_name).train(eta=0.001,
                                                          epochs=2000)
# model.test(dataset='mnist')

# %%

# E6 Tests

k, t = 10, 50
# Compute K Nearest Neighbors
knn = NearestNeighbors(n_neighbors=k).fit(data)

file_name = "e6_" + str(k) + "_" + str(t)
model = GeneralizedAutoEncoder(data, labels, layers,
                               display_step=1000,
                               use_knn=True,
                               knn=knn,
                               k=k, t=t,
                               loss_function='e6',
                               file_name=file_name).train(eta=0.01,
                                                          epochs=2000)
# model.test(dataset='mnist')

# %%

# Cost function E1

model = GeneralizedAutoEncoder(data, labels, layers,
                               learning_rate=0.0001,
                               epochs=10000,
                               display_step=1000,
                               loss_function='e1',
                               file_name="gae_e1").train()
model.test(dataset='mnist')

# Cost Function E2

# Compute K Nearest Neighbors
knn = NearestNeighbors(n_neighbors=5).fit(data)
file_name = "e2_"
model = GeneralizedAutoEncoder(data, labels, layers,
                               learning_rate=0.0001,
                               epochs=10000,
                               display_step=1000,
                               use_knn=True,
                               knn=knn,
                               k=5, t=1,
                               loss_function='e2',
                               file_name=file_name).train()
model.test(dataset='mnist')


# %%

# Cost Function E1
model = GeneralizedAutoEncoder(data, labels, layers,
                               learning_rate=0.0001,
                               epochs=50000,
                               display_step=1000,
                               loss_function='e1',
                               file_name="gae_e1").train()
model.test(dataset='mnist')


# %%
# Cost Function E2
for i in [5, 10, 20, 50]:

    # Compute K Nearest Neighbors
    knn = NearestNeighbors(n_neighbors=i).fit(data)

    for j in [2, 8, 32, 72, 128, 200]:

        file_name = "e2_" + str(i) + "_" + str(j)
        model = GeneralizedAutoEncoder(data, labels, layers,
                                       learning_rate=0.0001,
                                       epochs=50000,
                                       display_step=1000,
                                       use_knn=True,
                                       knn=knn,
                                       k=i, t=j,
                                       loss_function='e2',
                                       file_name=file_name).train()
        model.test(dataset='mnist')


# %%
# Cost Function E3
for i in [5, 10, 20, 50]:

    # Compute K Nearest Neighbors
    knn = NearestNeighbors(n_neighbors=i).fit(data)

    for j in [2, 8, 32, 72, 128, 200]:

        file_name = "e3_" + str(i) + "_" + str(j)
        model = GeneralizedAutoEncoder(data, labels, layers,
                                       learning_rate=0.0001,
                                       epochs=50000,
                                       display_step=1000,
                                       use_knn=True,
                                       knn=knn,
                                       k=i, t=j,
                                       loss_function='e3',
                                       file_name=file_name).train()
        model.test(dataset='mnist')


# %%
# Cost Function E4
for i in [15]:

    # Compute K Nearest Neighbors
    knn = NearestNeighbors(n_neighbors=i).fit(data)

    for j in [8, 72]:

        file_name = "e4_" + str(i) + "_" + str(j)
        model = GeneralizedAutoEncoder(data, labels, layers,
                                       learning_rate=0.0001,
                                       epochs=50000,
                                       display_step=1000,
                                       use_knn=True,
                                       knn=knn,
                                       k=i, t=j,
                                       lmbda=0.1,
                                       loss_function='e4',
                                       file_name=file_name).train()
        model.test(dataset='mnist')


# %%
# Cost Function E5
for i in [15]:

    # Compute K Nearest Neighbors
    knn = NearestNeighbors(n_neighbors=i).fit(data)

    for j in [8, 72]:

        file_name = "e5_" + str(i) + "_" + str(j)
        model = GeneralizedAutoEncoder(data, labels, layers,
                                       learning_rate=0.0001,
                                       epochs=50000,
                                       display_step=1000,
                                       use_knn=True,
                                       knn=knn,
                                       k=i, t=j,
                                       loss_function='e5',
                                       file_name=file_name).train()
        model.test(dataset='mnist')


# %%
# Cost Function E6
for i in [15]:

    # Compute K Nearest Neighbors
    knn = NearestNeighbors(n_neighbors=i).fit(data)

    for j in [8, 72]:

        file_name = "e6_" + str(i) + "_" + str(j)
        model = GeneralizedAutoEncoder(data, labels, layers,
                                       learning_rate=0.0001,
                                       epochs=50000,
                                       display_step=1000,
                                       use_knn=True,
                                       knn=knn,
                                       k=i, t=j,
                                       loss_function='e6',
                                       file_name=file_name).train()
        model.test(dataset='mnist')


# %%

n_cluster = 10
kmeans = KMeans(n_cluster).fit(data)
data = np.concatenate((data, kmeans.cluster_centers_))

# Cost Function E8
for i in [15]:

    # Compute K Nearest Neighbors, exclude appended cluster centers
    knn = NearestNeighbors(n_neighbors=i).fit(data[:-n_cluster])

    for j in [8, 72]:

        file_name = "e8_" + str(i) + "_" + str(j)
        model = GeneralizedAutoEncoder(data, labels, layers,
                                       learning_rate=0.0001,
                                       epochs=50000,
                                       display_step=1000,
                                       use_knn=True,
                                       knn=knn,
                                       k=i, t=j,
                                       alpha=0.1, lmbda=0.1,
                                       n_cluster=n_cluster,
                                       use_kmeans=True,
                                       kmeans=kmeans,
                                       loss_function='e8',
                                       file_name=file_name).train()
        model.test(dataset='mnist')


# %%

n_cluster = 10
fcm = FCM(data, c=n_cluster, m=1.1, max_iter=10000).fit()
data = np.concatenate((data, fcm.cluster_centers))

# Cost Function E9
for i in [15]:

    # Compute K Nearest Neighbors, exclude appended cluster centers
    knn = NearestNeighbors(n_neighbors=i).fit(data[:-n_cluster])

    for j in [8, 72]:

        file_name = "e9_" + str(i) + "_" + str(j)
        model = GeneralizedAutoEncoder(data, labels, layers,
                                       learning_rate=0.0001,
                                       epochs=50000,
                                       display_step=1000,
                                       t=j, n_cluster=10,
                                       use_fcm=True,
                                       fcm=fcm,
                                       loss_function='e9',
                                       file_name=file_name).train()
        model.test(dataset='mnist')


# %%

# 1. To use geodesic weights
kng = knn.kneighbors_graph()
rw = graph_shortest_path(kng, method='auto', directed=False)


# %%

a = np.array([1, 2, 3, 0, 0, 4, 5, 0, 6])
b = np.where(a == 0)

v = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
        ])

print(np.sum(v, axis=0))
print(np.sum(v, axis=1))
                    

v1 = np.array([1, 2, 3])
v2 = np.array([3, 4 ,5])
v3 = np.array([5, 6, 7])




print(v1 + v2 + v3)

