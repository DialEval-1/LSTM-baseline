import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn_cell_impl import LSTMCell

MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.


def _sender_aware_encoding(inputs, senders):
    """ convert embedding into sender-aware embedding
    if sender is 1, the embedding (the output) is [0, 0, ...,  0, embed[0], embed[1],....]
    if sender is 0, the embedding is [embed[0], embed[1], ..., 0, 0, 0]
    """

    inputs_repeat = tf.tile(inputs, [1, 1, 2])
    mask_0 = tf.tile(tf.expand_dims(tf.logical_not(senders), -1), [1, 1, tf.shape(inputs)[-1]])
    mask_1 = tf.tile(tf.expand_dims(senders, -1), [1, 1, tf.shape(inputs)[-1]])
    mask = tf.concat([mask_0, mask_1], axis=-1)
    output = tf.where(mask, inputs_repeat, tf.zeros_like(inputs_repeat))
    return output


def _rnn(inputs, seq_lengths, dropout, params):
    cell = rnn_cell.DropoutWrapper(
        rnn_cell.MultiRNNCell(
            [LSTMCell(params.hidden_size) for _ in range(params.num_layer)]
        ),
        output_keep_prob=1 - dropout,
        variational_recurrent=True,
        dtype=tf.float32
    )

    output, hidden = tf.nn.dynamic_rnn(cell, inputs, sequence_length=seq_lengths, dtype=inputs.dtype)
    return output, hidden


def _encoder(inputs, senders, dialog_lengths, dropout, params):
    inputs = tf.reduce_sum(inputs, axis=2)            # Bag of Words
    inputs = _sender_aware_encoding(inputs, senders)  # [batch_num, dialog_len, (2 x embedding_size)]
    return _rnn(inputs, dialog_lengths, dropout, params)


def quality_model_fn(features, dropout, params):
    inputs, senders, utterance_lengths, dialog_lengths = features
    output, hidden = _encoder(inputs, senders, dialog_lengths, dropout, params)
    return tf.squeeze(tf.layers.dense(hidden[-1].h, 1))


def nugget_model_fn(features, dropout, params):
    inputs, senders, utterance_lengths, dialog_lengths = features
    output, hidden = _encoder(inputs, senders, dialog_lengths, dropout, params)

    # assume ordering is  [customer, helpdesk, customer, .....]
    customer_index = tf.range(start=0, delta=2, limit=tf.shape(output)[1])
    helpdesk_index = tf.range(start=1, delta=2, limit=tf.shape(output)[1])

    customer_output = tf.gather(output, indices=customer_index, axis=1)
    helpdesk_output = tf.gather(output, indices=helpdesk_index, axis=1)

    customer_logits = tf.layers.dense(customer_output, 4)
    helpdesk_logits = tf.layers.dense(helpdesk_output, 3)

    return customer_logits, helpdesk_logits


def nugget_loss(customer_logits, helpdesk_logits, customer_labels, helpdesk_labels, dialogue_length):
    mask = tf.sequence_mask(dialogue_length)

    customer_index = tf.range(start=0, delta=2, limit=tf.shape(dialogue_length)[1])
    helpdesk_index = tf.range(start=1, delta=2, limit=tf.shape(dialogue_length)[1])

    customer_mask = tf.gather(mask, indices=customer_index, axis=1)
    helpdesk_mask = tf.gather(mask, indices=helpdesk_index, axis=1)

    customer_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=customer_logits, labels=customer_labels)
    customer_loss = tf.reduce_sum(tf.where(customer_mask, customer_loss, tf.zeros_like(customer_loss))) \
                    / tf.cast(tf.shape(customer_logits)[0], dtype=tf.float32)

    helpdesk_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=helpdesk_logits, labels=helpdesk_labels)
    helpdesk_loss = tf.reduce_sum(tf.where(helpdesk_mask, helpdesk_loss, tf.zeros_like(helpdesk_loss))) \
                    / tf.cast(tf.shape(helpdesk_logits)[0], dtype=tf.float32)

    return helpdesk_loss + customer_loss


def quality_loss(predictions, labels):
    return tf.losses.mean_squared_error(predictions=predictions,
                                        labels=labels,
                                        reduction=tf.losses.Reduction.MEAN)


def train(loss, global_step, lr=None):
    if lr is not None:
        opt = tf.train.AdadeltaOptimizer(lr)
    else:
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
