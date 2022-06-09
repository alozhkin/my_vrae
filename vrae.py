from random import random, Random

import keras
import numpy as np
from keras import layers, Input, Model
from keras import backend as K
from keras.callbacks import LambdaCallback

# Seed value
# Apparently you may use different seed values at each stage
seed_value= 0

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value)
# for later versions:
# tf.compat.v1.set_random_seed(seed_value)

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
    kl = 0.5 * K.sum(K.exp(z_log_sigma) + K.square(z_mean) - 1. - z_log_sigma)
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
     # 'result': km_pred
     'result': centroid_idxs
     })

# plt.hist(a, 9)
# plt.show()

# fig, ax = plt.subplots()
# plt.figure(figsize=(10,8))
# bins = np.linspace(0, 8, 40)
# for ii in a['result'].unique():
#     subset = a[a['expected'] == ii]['result'].tolist()
#     ax.hist(subset, bins=9, alpha=0.5, label=f"Cluster {ii}")
# ax.legend()
# plt.show()

a = pd.crosstab(a['expected'], a['result'])
print(a)

a.plot.bar(stacked=True)
