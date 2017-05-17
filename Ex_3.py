from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np

from keras import backend as K
from keras.engine import InputSpec
from keras.engine.topology import Layer
from keras.layers.recurrent import LSTM
from keras.models import Sequential

import theano
from theano import tensor as T

class MixtureDensity(Layer):
   """
   Produces, for each mixture component:
      mean: u <= R
      standard deviation: o <= R+
      weight: w <= [0, 1], sum(Wi) = 1
   """
   def __init__(self, mixture_components, **kwargs):
      self.mixture_components = mixture_components
      self.input_spec = [InputSpec(ndim=2)]
      super(MixtureDensity, self).__init__(**kwargs)

   def call(self, x, mask=None):
      x = T.set_subtensor(
         x[:, self.mixture_components: 2 * self.mixture_components],
          T.exp(x[:, self.mixture_components: 2 * self.mixture_components]))
      x = T.set_subtensor(
         x[:, 2 * self.mixture_components: 3 * self.mixture_components],
         T.nnet.softmax(x[:, 2 * self.mixture_components: 3 * self.mixture_components]))
      return x

   def build(self, input_shape):
      assert len(input_shape) == 2
      input_dim = input_shape[1]
      self.input_spec = [InputSpec(dtype=K.floatx(),
                                   shape=(None, input_dim))]

   def get_output_shape_for(self, input_shape):
      assert input_shape and len(input_shape) == 2
      return input_shape


def mixture_density_log_probability_error(y_true, y_pred):
   root_two_pi = np.sqrt(2 * np.pi)

   """
   :param y_true: R^1: best motor position
   :param y_pred: R^3M: parameters for mixture distribution (means, standard deviations, weights
   :return: negative log probability of y_true given parameters y_pred
   """
   def mixture_component_loss(component_index, components, y_true, y_pred):
      mu = y_pred[:, component_index]
      sigma = y_pred[:, components + component_index]
      alpha = y_pred[:, 2 * components + component_index]
      return (alpha / (sigma * root_two_pi)) * T.exp(-T.sqr(mu - y_true) / (2 * sigma ** 2))

   mixture_components = T.shape(y_pred)[1] / 3
   mixture_component_indices = T.arange(mixture_components)
   scan_result, _ = theano.scan(
      fn = mixture_component_loss,
      outputs_info=None,
      sequences=mixture_component_indices,
      non_sequences=[mixture_components, y_true, y_pred])
   return -T.log(scan_result.sum(0))


# since we are using stateful rnn tsteps can be set to 1
tsteps = 1
batch_size = 1
epochs = 25
# number of elements ahead that are used to make the prediction
lahead = 100
# number of mixture components
mixture_components = 3


def gen_cosine_amp(amp=100, period=25, x0=0, xn=50000, step=1, k=0.0001):
    """Generates an absolute cosine time series with the amplitude
    exponentially decreasing

    Arguments:
        amp: amplitude of the cosine function
        period: period of the cosine function
        x0: initial x of the time series
        xn: final x of the time series
        step: step of the time series discretization
        k: exponential rate
    """
    cos = np.zeros(((xn - x0) * step, 1, 1))
    for i in range(len(cos)):
        idx = x0 + i * step
        cos[i, 0, 0] = amp * np.cos(idx / (2 * np.pi * period))
        cos[i, 0, 0] = cos[i, 0, 0] * np.exp(-k * idx)
    return cos


print('Generating Data')
cos = gen_cosine_amp()
print('Input shape:', cos.shape)

expected_output = np.zeros((len(cos), 1))
for i in range(len(cos) - lahead):
    expected_output[i, 0] = np.mean(cos[i + 1:i + lahead + 1])

print('Output shape')
print(expected_output.shape)

print('Creating Model')
model = Sequential()

model.add(
   LSTM(
      mixture_components * 3,
      batch_input_shape=(batch_size, tsteps, 1),
      stateful=True))

model.add(MixtureDensity(mixture_components))

model.compile(
   loss=mixture_density_log_probability_error,
   optimizer='rmsprop')

summary = model.summary()

config = model.get_config()

print('Training')
for i in range(epochs):
    print('Epoch', i, '/', epochs)
    model.fit(cos,
              expected_output,
              batch_size=batch_size,
              verbose=1,
              nb_epoch=1,
              shuffle=False)
    model.reset_states()

print('Predicting')
predicted_output = model.predict(cos, batch_size=batch_size)

print('Ploting Results')
plt.subplot(2, 1, 1)
plt.plot(expected_output)
plt.title('Expected')
plt.subplot(2, 1, 2)
plt.plot(predicted_output)
plt.title('Predicted')
plt.show()
