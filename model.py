import numpy as np
import tensorflow as tf
from pathlib2 import Path
from tensorflow.contrib.rnn import stack_bidirectional_dynamic_rnn
from tensorflow.python.ops import rnn_cell

import data
from data import Task


class Model(object):
    def __init__(self, embedding, task, params, session=None, graph=None):
        self.graph = graph or tf.get_default_graph()
        self.session = session or tf.Session()
        self.task = task
        self.turns = tf.placeholder(
            shape=(None, None, None), dtype=tf.int32, name="turns")
        self.senders = tf.placeholder(
            shape=(None, None), dtype=tf.bool, name="senders")
        self.turn_lengths = tf.placeholder(
            shape=(None, None), dtype=tf.int32, name="turn_lengths")
        self.dialogue_lengths = tf.placeholder(
            shape=(None), dtype=tf.int32, name="dialogue_lengths")
        self.h_nuggets_labels = tf.placeholder(
            shape=(None, None, len(data.HELPDESK_NUGGET_TYPES_WITH_PAD)),
            dtype=tf.float32, name="helpdesk_nuggets_labels")
        self.c_nuggets_labels = tf.placeholder(
            shape=(None, None, len(data.CUSTOMER_NUGGET_TYPES_WITH_PAD)),
            dtype=tf.float32, name="customer_nuggets_labels")
        self.quality_labels = tf.placeholder(
            shape=(None, len(data.QUALITY_MEASURES), len(data.QUALITY_SCALES)),
            dtype=tf.float32, name="quality_labels")
        self.dropout = tf.placeholder_with_default(params.dropout, shape=[], name="dropout_rate")

        with tf.device("/cpu:0"):
            self.embedding = tf.get_variable(shape=embedding.shape, trainable=params.update_embedding,
                                             name="embedding_weights", dtype=tf.float32)
            turns_embedded = tf.nn.embedding_lookup(self.embedding, self.turns)
        turns_boW = tf.reduce_sum(turns_embedded, axis=2, name="BoW")  # Bag of Words

        features = (turns_boW, self.senders, self.turn_lengths, self.dialogue_lengths)

        if task == Task.nugget:
            self.c_nuggets_logits, self.h_nuggets_logits = nugget_model_fn(features, self.dropout, params)
            self.loss = nugget_loss(
                self.c_nuggets_logits, self.h_nuggets_logits,
                self.c_nuggets_labels, self.h_nuggets_labels, self.dialogue_lengths, tf.shape(self.turns)[1])

            self.prediction = (tf.nn.softmax(self.c_nuggets_logits, axis=-1),
                               tf.nn.softmax(self.h_nuggets_logits, axis=-1))

        elif task == Task.quality:
            self.quality_logits = quality_model_fn(features, self.dropout, params)
            self.loss = quality_loss(self.quality_logits, self.quality_labels)
            self.prediction = (tf.nn.softmax(self.quality_logits, axis=-1))

        else:
            raise ValueError("Unexpected Task: %s" % task.name)

        self.train_op = build_train_op(self.loss, tf.train.get_or_create_global_step(),
                                       lr=params.learning_rate, optimizer=params.optimizer)

        # embedding weights are not saved into Graph if it is not trainable
        self.saver = tf.train.Saver(tf.trainable_variables(), save_relative_paths=True)

        self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) if params.trace else tf.RunOptions()
        self.run_metadata = tf.RunMetadata()

        self.session.run(tf.global_variables_initializer())
        # Variable init does not accept weights that are larger than 2GB
        # This must be ran after global_variables_initializer
        self.session.run(self.embedding.initializer, feed_dict={self.embedding.initial_value: embedding})

        assert np.allclose(self.session.run(self.embedding), embedding)

    def save_model(self, save_path: Path=None):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.saver.save(self.session, str(save_path))

    def load_model(self, restore_path=None):
        path =tf.train.latest_checkpoint(str(restore_path))
        self.saver.restore(self.session, path)


    def train_batch(self, batch_op):
        (_, turns, senders, turn_lengths, dialog_lengths,
         c_nugget_labels, h_nugget_labels, quality_labels) = self.session.run(batch_op)

        feed_dict = {
            self.turns: turns,
            self.senders: senders,
            self.turn_lengths: turn_lengths,
            self.dialogue_lengths: dialog_lengths,
            self.c_nuggets_labels: c_nugget_labels,
            self.h_nuggets_labels: h_nugget_labels,
            self.quality_labels: quality_labels,
        }

        _, loss = self.session.run(
            [self.train_op, self.loss],
            feed_dict=feed_dict,
            run_metadata=self.run_metadata,
            options=self.run_options
        )

        return loss

    def train_epoch(self, batch_initializer, batch_op, n_epoch=1,
                    reduce_fn=np.mean, save_path=None, save_per_epoch=True):
        results = []
        for i in range(n_epoch):
            self.session.run(batch_initializer)
            while True:
                try:
                    results.append(self.train_batch(batch_op))
                    if save_per_epoch and save_path:
                        self.save_model(save_path)
                except tf.errors.OutOfRangeError:
                    break
        return reduce_fn(results)

    def __predict_batch(self, batch_op):
        (dialog_ids, turns, senders, turn_lengths, dialog_lengths) = self.session.run(batch_op)

        feed_dict = {
            self.turns: turns,
            self.senders: senders,
            self.turn_lengths: turn_lengths,
            self.dialogue_lengths: dialog_lengths,
            self.dropout: 0.,
        }

        outputs = self.session.run(self.prediction, feed_dict=feed_dict)

        if isinstance(outputs, tuple):
            return dialog_ids, zip(*outputs), dialog_lengths
        return dialog_ids, outputs, dialog_lengths

    def predict(self, batch_initializer, batch_op):
        results = []
        self.session.run(batch_initializer)
        while True:
            try:
                results.extend(zip(*self.__predict_batch(batch_op)))
            except tf.errors.OutOfRangeError:
                break
        return results


def _sender_aware_encoding(inputs, senders):
    """ convert embedding into sender-aware embedding
    if sender is 1, the embedding (the output) is [0, 0, ...,  0, embed[0], embed[1],....]
    if sender is 0, the embedding is [embed[0], embed[1], ..., 0, 0, 0]
    """
    with tf.name_scope("sender_aware_encoding"):
        inputs_repeat = tf.tile(inputs, [1, 1, 2])
        mask_0 = tf.tile(tf.expand_dims(tf.logical_not(senders), -1), [1, 1, tf.shape(inputs)[-1]])
        mask_1 = tf.tile(tf.expand_dims(senders, -1), [1, 1, tf.shape(inputs)[-1]])
        mask = tf.concat([mask_0, mask_1], axis=-1)
        output = tf.where(mask, inputs_repeat, tf.zeros_like(inputs_repeat))
        return output


def _rnn(inputs, seq_lengths, dropout, params):
    with tf.name_scope("rnn"):
        cell_fn = lambda: rnn_cell.DropoutWrapper(
            params.cell(params.hidden_size),
            output_keep_prob=1 - dropout,
            variational_recurrent=True,
            dtype=tf.float32
        )

        fw_cells = [cell_fn() for _ in range(params.num_layers)]
        bw_cells = [cell_fn() for _ in range(params.num_layers)]

        output, _, _ = stack_bidirectional_dynamic_rnn(fw_cells, bw_cells, inputs, sequence_length=seq_lengths,
                                                       dtype=tf.float32)

        return output


def _encoder(inputs, senders, dialog_lengths, dropout, params):
    inputs = _sender_aware_encoding(inputs, senders)  # [batch_num, dialog_len, (2 x embedding_size)]
    return _rnn(inputs, dialog_lengths, dropout, params)


def quality_model_fn(features, dropout, params):
    turns, senders, utterance_lengths, dialog_lengths = features
    output = _encoder(turns, senders, dialog_lengths, dropout, params)
    dialog_repr = tf.reduce_sum(output, axis=1)
    logits = []
    for _ in data.QUALITY_MEASURES:
        logits.append(tf.layers.dense(dialog_repr, len(data.QUALITY_SCALES)))
    logits = tf.stack(logits, axis=1)
    return logits


def nugget_model_fn(features, dropout, params):
    turns, senders, utterance_lengths, dialog_lengths = features
    output = _encoder(turns, senders, dialog_lengths, dropout, params)


    # assume ordering is  [customer, helpdesk, customer, .....]
    max_time = tf.shape(output)[1]
    customer_index = tf.range(start=0, delta=2, limit=max_time)
    helpdesk_index = tf.range(start=1, delta=2, limit=max_time)

    customer_output = tf.gather(output, indices=customer_index, axis=1)
    helpdesk_output = tf.gather(output, indices=helpdesk_index, axis=1)

    assert_op = tf.assert_equal(tf.shape(customer_output)[1] + tf.shape(helpdesk_output)[1], max_time)

    with tf.control_dependencies([assert_op]):
        customer_logits = tf.layers.dense(customer_output, len(data.CUSTOMER_NUGGET_TYPES_WITH_PAD))
        helpdesk_logits = tf.layers.dense(helpdesk_output, len(data.HELPDESK_NUGGET_TYPES_WITH_PAD))

    return customer_logits, helpdesk_logits


def nugget_loss(customer_logits, helpdesk_logits, customer_labels, helpdesk_labels, dialogue_lengths, max_dialogue_len):
    mask = tf.sequence_mask(dialogue_lengths)

    customer_index = tf.range(start=0, delta=2, limit=max_dialogue_len)
    helpdesk_index = tf.range(start=1, delta=2, limit=max_dialogue_len)

    customer_mask = tf.gather(mask, indices=customer_index, axis=1)
    helpdesk_mask = tf.gather(mask, indices=helpdesk_index, axis=1)

    customer_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=customer_logits, labels=customer_labels)
    customer_loss = tf.reduce_sum(tf.where(customer_mask, customer_loss, tf.zeros_like(customer_loss))) \
                    / tf.cast(tf.shape(customer_logits)[0], dtype=tf.float32)

    helpdesk_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=helpdesk_logits, labels=helpdesk_labels)
    helpdesk_loss = tf.reduce_sum(tf.where(helpdesk_mask, helpdesk_loss, tf.zeros_like(helpdesk_loss))) \
                    / tf.cast(tf.shape(helpdesk_logits)[0], dtype=tf.float32)

    return helpdesk_loss + customer_loss


def quality_loss(logits, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))


def build_train_op(loss, global_step, optimizer=None, lr=None, moving_decay=0.9999):
    if lr is not None:
        opt = optimizer(lr)
    else:
        opt = optimizer()
    grads = opt.compute_gradients(loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
    return train_op
