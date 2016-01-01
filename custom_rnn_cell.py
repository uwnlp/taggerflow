import tensorflow as tf
from tensorflow.python.ops import rnn_cell

class DyerLSTMCell(rnn_cell.RNNCell):
  """LSTM recurrent network cell variant from https://github.com/clab/cnn.
  Forgot and input gates are coupled.
  Gates contain peephole connections.
  """

  def __init__(self, num_units, input_size, freeze):
    self._num_units = num_units
    self._input_size = input_size
    self._freeze = freeze

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
    with tf.variable_scope(scope or type(self).__name__):  # "DyerLSTMCell"
      c, h = tf.split(1, 2, state)

      input_gate = tf.sigmoid(linear([inputs, h, c], self._num_units, "input_gate", freeze=self._freeze))
      new_input = tf.tanh(linear([inputs, h], self._num_units, "new_input", freeze=self._freeze))
      new_c = input_gate * new_input + (1.0 - input_gate) * c
      output_gate = tf.sigmoid(linear([inputs, h, new_c], self._num_units, "output_gate", freeze=self._freeze))
      new_h = tf.tanh(new_c) * output_gate

    return new_h, tf.concat(1, [new_c, new_h])

def linear(args, output_size, scope, freeze):
  """Slight modification of the linear function in rnn_cell."""

  assert args
  if not isinstance(args, (list, tuple)):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  # Now the computation.
  with tf.variable_scope(scope):
    matrix = maybe_get_variable("Matrix", [total_arg_size, output_size], freeze=freeze)
    bias = maybe_get_variable("Bias", [output_size], initializer=tf.constant_initializer(0.0), freeze=freeze)
  return tf.matmul(args[0] if len(args) == 1 else tf.concat(1, args), matrix) + bias

def maybe_get_variable(name, shape, initializer=None, freeze=False):
  variable = tf.get_variable(name, shape, initializer=initializer)
  if freeze:
    return tf.get_default_session().run(variable)
  else:
    return variable
