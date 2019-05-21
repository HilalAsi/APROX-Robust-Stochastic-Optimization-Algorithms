# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Implementation of the optimization methods (Truncated, Truncated-Adagrad)
from the paper https://arxiv.org/abs/1903.08619
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from six.moves import zip  # pylint: disable=redefined-builtin

import tensorflow as tf
from tensorflow.python import tf2
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizer_v2 import adadelta as adadelta_v2
from tensorflow.python.keras.optimizer_v2 import adagrad as adagrad_v2
from tensorflow.python.keras.optimizer_v2 import adam as adam_v2
from tensorflow.python.keras.optimizer_v2 import adamax as adamax_v2
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_v2
from tensorflow.python.keras.optimizer_v2 import nadam as nadam_v2
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.optimizer_v2 import rmsprop as rmsprop_v2
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer as tf_optimizer_module
from tensorflow.python.training import training_util
from tensorflow.python.training.checkpointable import base as checkpointable
from tensorflow.python.util.tf_export import tf_export


from tensorflow.keras.optimizers import Optimizer

#@tf_export(v1=['keras.optimizers.Truncated'])
class Truncated(Optimizer):
  """Stochastic truncated gradient optimizer.
  Includes support for momentum,
  learning rate decay, and Nesterov momentum.
  Arguments:
      lr: float >= 0. Learning rate.
      momentum: float >= 0. Parameter that accelerates the algorithm
          in the relevant direction and dampens oscillations.
      decay: float >= 0. Learning rate decay over each update.
      nesterov: boolean. Whether to apply Nesterov momentum.
  
  References
      - [The importance of better models in stochastic optimization](https://arxiv.org/abs/1903.08619)
  
  """

  def __init__(self, lr=0.01, momentum=0., decay=0., nesterov=False, **kwargs):
    super(Truncated, self).__init__(**kwargs)
    with K.name_scope(self.__class__.__name__):
      self.iterations = K.variable(0, dtype='int64', name='iterations')
      self.lr = K.variable(lr, name='lr')
      self.momentum = K.variable(momentum, name='momentum')
      self.decay = K.variable(decay, name='decay')
    self.initial_decay = decay
    self.nesterov = nesterov

  def get_updates(self, loss, params):
    grads = self.get_gradients(loss, params)
    self.updates = [state_ops.assign_add(self.iterations, 1)]

    lr = self.lr
    if self.initial_decay > 0:
      lr = lr * (  # pylint: disable=g-no-augmented-assignment
          1. / (1. + self.decay * math_ops.cast(self.iterations,
                                                K.dtype(self.decay))))

    #calculate truncated step size
    global_grad_norm = tf.global_norm(grads)**2 #TODO verify this
    lr_trunc = tf.cond(0 < global_grad_norm, lambda: tf.divide(loss, global_grad_norm), lambda: lr)
    lr = tf.minimum(lr,lr_trunc)
	
	# momentum
    shapes = [K.int_shape(p) for p in params]
    moments = [K.zeros(shape) for shape in shapes]
    self.weights = [self.iterations] + moments
    for p, g, m in zip(params, grads, moments):
      v = self.momentum * m - lr * g  # velocity
      self.updates.append(state_ops.assign(m, v))

      if self.nesterov:
        new_p = p + self.momentum * v - lr * g
      else:
        new_p = p + v

      # Apply constraints.
      if getattr(p, 'constraint', None) is not None:
        new_p = p.constraint(new_p)

      self.updates.append(state_ops.assign(p, new_p))
    return self.updates

  def get_config(self):
    config = {
        'lr': float(K.get_value(self.lr)),
        'momentum': float(K.get_value(self.momentum)),
        'decay': float(K.get_value(self.decay)),
        'nesterov': self.nesterov
    }
    base_config = super(Truncated, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
	
	
	
	
	
	
	
	
	
#@tf_export(v1=['keras.optimizers.TruncatedAdagrad'])
class TruncatedAdagrad(Optimizer):
  """Implementation of the TruncatedAdagrad optimizer.

  Arguments
      lr: float >= 0. Initial learning rate.
      epsilon: float >= 0. If `None`, defaults to `K.epsilon()`.
      decay: float >= 0. Learning rate decay over each update.
  References
      - [The importance of better models in stochastic optimization](https://arxiv.org/abs/1903.08619)
  """

  def __init__(self, lr=0.01, epsilon=None, decay=0., **kwargs):
    super(TruncatedAdagrad, self).__init__(**kwargs)
    with K.name_scope(self.__class__.__name__):
      self.lr = K.variable(lr, name='lr')
      self.decay = K.variable(decay, name='decay')
      self.iterations = K.variable(0, dtype='int64', name='iterations')
    if epsilon is None:
      epsilon = K.epsilon()
    self.epsilon = epsilon
    self.initial_decay = decay

  def get_updates(self, loss, params):
    grads = self.get_gradients(loss, params)
    shapes = [K.int_shape(p) for p in params]
    accumulators = [K.zeros(shape) for shape in shapes]
    self.weights = accumulators
    self.updates = [state_ops.assign_add(self.iterations, 1)]

    lr = self.lr
    if self.initial_decay > 0:
      lr = lr * (  # pylint: disable=g-no-augmented-assignment
          1. / (1. + self.decay * math_ops.cast(self.iterations,
                                                K.dtype(self.decay))))

    #calculate the truncated-adagrad stepsize

    #global_grad_norm = tf.global_norm(grads)**2 #TODO verify this
    rotated_grad_norm = 0
    for i in range(0,len(grads)):
        rotated_grad_norm += tf.reduce_sum(tf.multiply(tf.truediv(grads[i],tf.sqrt(accumulators[i])+ self.epsilon),grads[i])) #TODO verify this
    lr_trunc = tf.cond(0 < rotated_grad_norm, lambda: tf.divide(loss, rotated_grad_norm), lambda: lr)
    lr = tf.minimum(lr, lr_trunc)
	
	
    for p, g, a in zip(params, grads, accumulators):
      new_a = a + math_ops.square(g)  # update accumulator
      self.updates.append(state_ops.assign(a, new_a))
      new_p = p - lr * g / (K.sqrt(new_a) + self.epsilon)

      # Apply constraints.
      if getattr(p, 'constraint', None) is not None:
        new_p = p.constraint(new_p)

      self.updates.append(state_ops.assign(p, new_p))
    return self.updates

  def get_config(self):
    config = {
        'lr': float(K.get_value(self.lr)),
        'decay': float(K.get_value(self.decay)),
        'epsilon': self.epsilon
    }
    base_config = super(TruncatedAdagrad, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))