import argparse
import os
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn_cell_impl import LSTMCell

PROJECT_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir))

parser = argparse.ArgumentParser()
parser.add_argument("--category", type=str, default="CS")
parser.add_argument("--hidden-size", type=int, default=150)
parser.add_argument("--num_layer", type=int, default=3)
parser.add_argument("--update_embedding", type=bool, default=False)

FLAGS = parser.parse_args()


def sender_aware_encoding(inputs, senders):
    """ convert embedding into sender-aware embedding
    if sender is 1, the embedding is [0, 0, ...,  0, embed[0], embed[1],....]
    if sender is 0, the embedding is [embed[0], embed[1], ..., 0, 0, 0]
    """

    inputs_repeat = tf.tile(inputs, [1, 1, 2])
    mask_0 = tf.tile(tf.expand_dims(tf.logical_not(senders), -1), [1, 1, tf.shape(inputs)[-1]])
    mask_1 = tf.tile(tf.expand_dims(senders, -1), [1, 1, tf.shape(inputs)[-1]])
    mask = tf.concat([mask_0, mask_1], axis=-1)
    output = tf.where(mask, inputs_repeat, tf.zeros_like(inputs_repeat))
    return output


def rnn_encoder(inputs, seq_lengths, dropout_keep_rate):
    cell = rnn_cell.DropoutWrapper(
        rnn_cell.MultiRNNCell(
            [LSTMCell(FLAGS.hidden_size) for _ in range(FLAGS.num_layer)]
        ),
        output_keep_prob=dropout_keep_rate,
        variational_recurrent=True,
        state_keep_prob=dropout_keep_rate,
        dtype=tf.float32
    )

    output, hidden = tf.nn.dynamic_rnn(cell, inputs, sequence_length=seq_lengths, dtype=inputs.dtype)
    return hidden[-1].h


def inference(inputs, senders, utterance_lengths, dialog_lengths,  dropout_keep_rate):


    inputs = tf.reduce_sum(inputs, axis=2)
    inputs = sender_aware_encoding(inputs, senders)  # [batch_num, dialog_len, (2 x embedding_size)]
    inputs = rnn_encoder(inputs, dialog_lengths, dropout_keep_rate)  # [batch_num, hidden_size]
    inputs = tf.layers.dropout(inputs)
    return tf.reshape(tf.layers.dense(inputs, 1), [-1])


def calculate_loss(predictions, labels):
    return tf.losses.mean_squared_error(predictions=predictions,
                                        labels=labels,
                                        reduction=tf.losses.Reduction.MEAN)

MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.

def train(loss, global_step):
    opt = tf.train.AdamOptimizer()
    grads = opt.compute_gradients(loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
    return train_op
