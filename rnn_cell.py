from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np

class RNNCell(tf.nn.rnn_cell.RNNCell):
    """Wrapper around our RNN cell implementation that allows us to play
    nicely with TensorFlow.
    """
    def __init__(self, input_size, state_size):
        self.input_size = input_size
        self._state_size = state_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def __call__(self, inputs, state, scope=None):
        """Updates the state using the previous @state and @inputs.
        Remember the RNN equations are:

        h_t = sigmoid(x_t W_x + h_{t-1} W_h + b)

        TODO: In the code below, implement an RNN cell using @inputs
        (x_t above) and the state (h_{t-1} above).
            - Define W_x, W_h, b to be variables of the apporiate shape
              using the `tf.get_variable' functions. Make sure you use
              the names "W_x", "W_h" and "b"!
            - Compute @new_state (h_t) defined above
        Tips:
            - Remember to initialize your matrices using the xavier
              initialization as before.
        Args:
            inputs: is the input vector of size [None, self.input_size]
            state: is the previous state vector of size [None, self.state_size]
            scope: is the name of the scope to be used when defining the variables inside.
        Returns:
            a pair of the output vector and the new state vector.
        """
        scope = scope or type(self).__name__

        # It's always a good idea to scope variables in functions lest they
        # be defined elsewhere!
        with tf.variable_scope(scope):
            W_h = tf.get_variable("W_h", 
                shape=[self.state_size, self.state_size], 
                initializer=tf.contrib.layers.xavier_initializer())
            W_x = tf.get_variable("W_x", 
                shape=[self.input_size, self.state_size], 
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b", 
                shape=[self.state_size], 
                initializer=tf.constant_initializer(0.0))
            z_x = tf.matmul(inputs, W_x)
            z_h = tf.add(tf.matmul(state, W_h), b)
            new_state = tf.nn.sigmoid(tf.add(z_x, z_h))
        # For an RNN , the output and state are the same (N.B. this
        # isn't true for an LSTM)
        output = new_state
        return output, new_state
