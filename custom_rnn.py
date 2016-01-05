import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import rnn

def fixed_rnn(cell, inputs, initial_state=None, dtype=None,
        sequence_length=None, scope=None):
  outputs = []
  states = []
  with tf.variable_scope(scope or "RNN"):
    batch_size = tf.shape(inputs[0])[0]
    if initial_state is not None:
      state = initial_state
    else:
      if not dtype:
        raise ValueError("If no initial_state is provided, dtype must be.")
      state = cell.zero_state(batch_size, dtype)

    if sequence_length:  # Prepare variables
      zero_output_state = (
          tf.zeros(tf.pack([batch_size, cell.output_size]),
                   inputs[0].dtype),
          tf.zeros(tf.pack([batch_size, cell.state_size]),
                   state.dtype))

      max_sequence_length = tf.reduce_max(sequence_length)

    for time, input_ in enumerate(inputs):
      if time > 0: tf.get_variable_scope().reuse_variables()
      def output_state():
        return cell(input_, state)
      if sequence_length:
        (output, state) = control_flow_ops.cond(
            time >= max_sequence_length,
            lambda: zero_output_state, output_state)
        output.set_shape([None, cell.output_size])
      else:
        (output, state) = output_state()

      outputs.append(output)
      states.append(state)

    return (outputs, states)

def bidirectional_rnn(cell_fw, cell_bw, inputs,
                      initial_state_fw=None, initial_state_bw=None,
                      dtype=None, sequence_length=None, scope=None):

  name = scope or "BiRNN"
  # Forward direction
  with tf.variable_scope(name + "_FW"):
    output_fw, _ = fixed_rnn(cell_fw, inputs, initial_state_fw, dtype, sequence_length)
  # Backward direction
  with tf.variable_scope(name + "_BW"):
    tmp, _ = fixed_rnn(
        cell_bw, rnn._reverse_seq(inputs, sequence_length), initial_state_bw, dtype, sequence_length)
  output_bw = rnn._reverse_seq(tmp, sequence_length)
  # Concat each of the forward/backward outputs
  outputs = [tf.concat(1, [fw, bw])
             for fw, bw in zip(output_fw, output_bw)]

  return outputs
