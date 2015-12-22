from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh

# So the code can easily be merged into rnn_cell.py.
from tensorflow.python.ops.rnn_cell import *

class DyerLSTMCell(RNNCell):
  """LSTM recurrent network cell variant from https://github.com/clab/cnn.
  Forgot and input gates are coupled.
  Gates contain peephole connections.
  """

  def __init__(self, num_units, input_size):
    self._num_units = num_units
    self._input_size = input_size

  @property
  def input_size(self):
    return self._input_size

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return 2 * self._num_units

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with vs.variable_scope(scope or type(self).__name__):  # "DyerLSTMCell"
      c, h = array_ops.split(1, 2, state)

      input_gate = sigmoid(linear([inputs, h, c], self._num_units, True, scope="input_gate"))
      new_input = tanh(linear([inputs, h], self._num_units, True, scope="new_input"))
      new_c = input_gate * new_input + (1.0 - input_gate) * c
      output_gate = sigmoid(linear([inputs, h, new_c], self._num_units, True, scope="output_gate"))
      new_h = tanh(new_c) * output_gate

    return new_h, array_ops.concat(1, [new_c, new_h])
