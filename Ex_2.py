import os
import theano
import numpy as np
import theano.tensor as T
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Layer, Dense, Lambda


def log_sum_exp(x, axis=None):
    x_max = T.max(x, axis=axis, keepdims=True)
    return T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=True)) + x_max

def neg_log_normal_mixture_likelihood(true, parameters):
    D = T.shape(true)[1]
    M = T.shape(parameters)[1] / (2 * D + 1)

    means = parameters[:, : D * M]
    sigmas = T.exp(parameters[:, D * M: 2 * D * M])
    weights = T.nnet.softmax(parameters[:, 2 * D * M:])
    two_pi = 2 * np.pi

    def component_normal_likelihood(i, mus, sis, als, tr):
        mu = mus[:, i * D:(i + 1) * D]
        sig = sis[:, i * D:(i + 1) * D]
        al = als[:, i]

        z = T.sum(((true - mu) / sig) ** 2, axis=1)/ -2.0
        normalizer = (T.prod(sig, axis=1) * two_pi)
        z += T.log(al) - T.log(normalizer)
        return z

    r, _ = theano.scan(
            fn=component_normal_likelihood,
            outputs_info=None,
            sequences=T.arange(M),
            non_sequences=[means, sigmas, weights, true])
    lp = log_sum_exp(r,0)
    loss = -T.mean(lp)
    return loss


base_dir = 'feats'
npy_files = [os.path.join(base_dir, f) for f in os.listdir(base_dir)]
data = []
for f in npy_files:
    t = np.load(f)
    data += list(t)
data = np.array(data)
data = data - data.mean()
data = data / data.std()

print('Data size:', data.shape)
M = 3
D = data.shape[1]

model = Sequential()
model.add(Dense(3, activation='sigmoid', input_shape=(D,)))
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.add(Dense(12, activation='sigmoid'))
model.add(Dense((2*D+1)*M, activation='linear'))
model.summary()
model.compile(loss=neg_log_normal_mixture_likelihood, optimizer='rmsprop')
model.fit(data, data, batch_size=1000, nb_epoch=100)
