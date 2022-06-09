from random import random, Random

import keras
import numpy as np
from keras import layers, Input, Model
from keras import backend as K

def remove_element(L, arr):
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind], arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        raise ValueError('array not found in list.')

seed_value = 42

import random

random.seed(seed_value)

import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering, OPTICS

users_num = 9
latent_dim = 9
intermediate_dim = 25
session_length, features = 50, 1


def sampling(args):
    z_mean, z_log_sigma = args
    batch_size = K.shape(z_mean)[0]
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.)
    return z_mean + z_log_sigma * epsilon


def vae_loss(input_x, decoder1, z_log_sigma, z_mean):
    recon = 0.00003 * K.sum(K.binary_crossentropy(input_x, decoder1))
    # recon = 0.003 * K.sqrt(K.mean(K.square(input_x - decoder1), axis=-1))
    kl = 0.05 * K.sum(K.exp(z_log_sigma) + K.square(z_mean) - 1. - z_log_sigma)
    # recon = K.print_tensor(recon)
    # kl = K.print_tensor(kl)
    return recon + kl


inputs = layers.Input(shape=(session_length, features), name='inputs')
encoder_rec = layers.LSTM(intermediate_dim, activation='relu', name='encoder_rec')(inputs)

z_mean = layers.Dense(latent_dim, name='z_mean')(encoder_rec)
z_log_sigma = layers.Dense(latent_dim, name='z_log_sigma')(encoder_rec)

sampled = layers.Lambda(sampling, name='sampled')([z_mean, z_log_sigma])

encoder = keras.Model(inputs, sampled, name='encoder')
mean_encoder = keras.Model(inputs, z_mean, name='encoder')
decoder_inputs = layers.RepeatVector(session_length, name='decoder_inputs')(sampled)
decoder_rec = layers.LSTM(intermediate_dim, activation='relu', return_sequences=True, name='decoder_rec')(
    decoder_inputs)
outputs = layers.TimeDistributed(layers.Dense(features, name='outputs'))(decoder_rec)
decoder = keras.Model(decoder_inputs, outputs, name='decoder')

m = Model(inputs, outputs, name='vrae')

m.add_loss(vae_loss(inputs, outputs, z_log_sigma, z_mean))
m.compile(loss=None, optimizer='adam')
# K.set_value(m.optimizer.learning_rate, 0.001)

# READ DATA AND TRAIN

df = pd.read_csv('/home/user/stuff/nn/lab2/my_vrae/resources/sessions.csv')

unique_vals = df['url'].unique()

websites_num = len(unique_vals)

df['url'].replace(to_replace=unique_vals,
                  value=list(range(len(unique_vals))),
                  inplace=True)

sessions = []
ids = []

for id in df['session_id'].unique():
    sessions.append(df[df['session_id'] == id]['url'].tolist())
    ids.append(df[df['session_id'] == id]['user_id'].tolist()[0])

temp = list(zip(sessions, ids))
Random(42).shuffle(temp)
sessions, ids = zip(*temp)

x = np.array([np.array(xi) for xi in sessions])
y = np.array(ids)

x_val = x[-200:]
y_val = y[-200:]
x_train = x[:-200]
y_train = y[:-200]

x_train, x_val = x_train / float(websites_num), x_val / float(websites_num)

# bb = np.average(x_train, axis=1).reshape(-1,1)

m.fit(x_train, x_train,
      epochs=20,
      batch_size=40,
      validation_data=(x_val, x_val))

m.save('/home/user/stuff/nn/lab2/my_vrae/resources')
# m = keras.models.load_model('/home/user/stuff/nn/lab2/my_vrae/resources')


xk = np.concatenate((x_train, x_val))
yk = np.concatenate((y_train, y_val))

# kmeans = OPTICS(min_samples=9)
kmeans = KMeans(n_clusters=users_num, n_init=20)
# kmeans = SpectralClustering(n_clusters=users_num, affinity='nearest_neighbors',
#                            assign_labels='kmeans')
latent_encoder_space = mean_encoder.predict(xk)
km_pred = kmeans.fit_predict(latent_encoder_space)
centers = kmeans.cluster_centers_

alpha = 1.0
max_iter = 300
iteration = 0
prev_centers = None

while np.not_equal(centers, prev_centers).any() and iteration < max_iter:
    sorted_points = [[] for _ in range(users_num)]
    for el in latent_encoder_space:
        dists = np.sqrt(np.sum((el - centers) ** 2, axis=1))
        centroid_idx = np.argmin(dists)
        sorted_points[centroid_idx].append(el)

    prev_centers = centers
    centers = [np.mean(cluster, axis=0) for cluster in sorted_points]
    for i, center in enumerate(centers):
        if np.isnan(center).any():
            centers[i] = prev_centers[i]

    iteration += 1

centroids = []
centroid_idxs = []
for el in latent_encoder_space:
    dists = np.sqrt(np.sum((el - centers) ** 2, axis=1))
    centroid_idx = np.argmin(dists)
    centroids.append(centers[centroid_idx])
    centroid_idxs.append(centroid_idx)

a = percentile_list = pd.DataFrame(
    {'expected': yk,
     'result': centroid_idxs
     })

a = pd.crosstab(a['expected'], a['result'])
print('\nVRAE+K-MEANS')
print(a)

a.plot.bar(stacked=True)


res = []

w = []

epsilon=0.05

centroid_X = np.average(xk, axis=1)

w.extend(centers)

sorted_points = [[] for _ in range(users_num)]

cluster_elements = []
for cluster in range(9):
    cluster_i = []
    cluster_elements.append(cluster_i)

cluster_lengths = np.zeros(9, dtype=int)

cluster_indices = []

for x in latent_encoder_space:
    dists = np.sqrt(np.sum((x - w) ** 2, axis=1))
    index = np.argmin(dists)
    # add cluster index of data x to a list
    cluster_indices.append(index)

    # update winner neuron
    w[index] = w[index] + 1 / (1 + cluster_lengths[index]) * (x - w[index])

    # append data to cluster
    cluster_elements[index].append(x)

    cluster_lengths[index] += 1


for epoch in range(20):
    loser = 0
    for i, x in enumerate(latent_encoder_space):
        dists = np.sqrt(np.sum((x - w) ** 2, axis=1))

        current_cluster_index = np.argmin(dists)

        x_th = i
        previous_cluster_index = cluster_indices[x_th]

        # check if current neuron is a loser
        if previous_cluster_index != current_cluster_index:
            # update winner neuron
            w[current_cluster_index] = w[current_cluster_index] + (x - w[current_cluster_index]) / (
                    cluster_lengths[current_cluster_index] + 1)

            # update loser neuron
            w[previous_cluster_index] = w[previous_cluster_index] - (x - w[previous_cluster_index]) / (
                    cluster_lengths[previous_cluster_index] - 1)

            # add and remove data to cluster
            cluster_elements[current_cluster_index] = list(cluster_elements[current_cluster_index])
            cluster_elements[current_cluster_index].append(x)
            remove_element(cluster_elements[previous_cluster_index], x)

            # update cluster index
            cluster_indices[x_th] = current_cluster_index

            cluster_lengths[current_cluster_index] += 1
            cluster_lengths[previous_cluster_index] -= 1

            loser += 1

    if loser == 0:
        break


a = percentile_list = pd.DataFrame(
    {'expected': yk,
     'result': cluster_indices
     })
a = pd.crosstab(a['expected'], a['result'])
print('\nCentNN')
print(a)

a.plot.bar(stacked=True)
