from datetime import datetime
import os

import time

from .utils import prepare_data, load_embedding
from data import dataset
import tensorflow as tf
import model as model
from runner import _SessionRunner
from scipy.stats import kendalltau
from scipy.stats import pearsonr
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent

model.parser.add_argument("--embedding-path", type=str, default=PROJECT_DIR / "data" /"embedding")
model.parser.add_argument("--data-path", type=str, default=PROJECT_DIR / "data")
model.parser.add_argument("--output-dir", type=str, default=PROJECT_DIR / "output/static")
model.parser.add_argument("--checkpoint-dir", type=str, default=PROJECT_DIR / "checkpoint/static")
model.parser.add_argument("--epoch-num", type=int, default=50)
model.parser.add_argument("--patience", type=int, default=5)
model.parser.add_argument("--batch-size", type=int, default=128)
model.parser.add_argument("--log-frequency", type=int, default=10)
model.parser.add_argument("--dropout", type=float, default=.8)
model.parser.add_argument("--test", type=bool, default=False)

FLAGS = model.parser.parse_args()


def main():
    with tf.Graph().as_default():
        train_path, dev_path, test_path = prepare_data(FLAGS.embedding_path, FLAGS.data_path)
        wtoi, itow, weight = load_embedding(FLAGS.embedding_path)
        train_dataset = dataset(train_path, wtoi, label_category=FLAGS.category, batch_size=FLAGS.batch_size)
        dev_dataset = dataset(dev_path, wtoi, label_category=FLAGS.category, batch_size=FLAGS.batch_size * 3)
        test_dataset = dataset(test_path, wtoi, label_category=FLAGS.category, batch_size=FLAGS.batch_size * 3)

        iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                                   train_dataset.output_shapes)

        training_init_op = iterator.make_initializer(train_dataset)
        dev_init_op = iterator.make_initializer(dev_dataset)
        test_init_op = iterator.make_initializer(test_dataset)

        with tf.device("/cpu:0"):

            dropout_keep_rate = tf.placeholder(dtype=tf.float32, shape=(), name="dropout_keep")
            batch = iterator.get_next()
            utterances, senders, utterance_length, dialog_length, labels = batch
            embedding_placeholder = tf.placeholder(tf.float32, weight.shape, )
            embedding = tf.Variable(tf.constant(0.0, shape=weight.shape), trainable=FLAGS.update_embedding,
                                    name="embedding", dtype=tf.float32)
            embedding_init = embedding.assign(embedding_placeholder)
            inputs = tf.nn.embedding_lookup(embedding,
                                            utterances)  # [batch_num, dialog_len, utterance_len, embedding_size]

        predictions = model.inference(inputs, senders, utterance_length,
                                      dialog_length, dropout_keep_rate)

        loss = model.calculate_loss(predictions, labels)
        train_op = model.train(loss, tf.train.get_or_create_global_step())

        class TrainSessionRunner(_SessionRunner):
            def after_run(self, extra_fetched_results):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = extra_fetched_results[0]
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), self._step, loss_value,
                                        examples_per_sec, sec_per_batch))

        saver = tf.train.Saver(max_to_keep=1)
        best_score = [[0, 0], [0, 0]]
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(embedding_init, feed_dict={embedding_placeholder: weight})

            train_helper = TrainSessionRunner(sess, [loss])
            no_improve = 0
            for epoch in range(FLAGS.epoch_num):
                if no_improve > FLAGS.patience:
                    print("Finish training %s, Best score:   Dev = %.4f (%f) test = %.4f(%f)" % (FLAGS.category,
                        best_score[0][0], best_score[0][1], best_score[1][0], best_score[1][1]))
                    return

                sess.run(training_init_op)
                while True:
                    try:
                        train_helper.run([train_op], feed_dict={dropout_keep_rate: FLAGS.dropout})
                    except tf.errors.OutOfRangeError:
                        break

                sess.run(dev_init_op)
                label_values_list = []
                pred_values_list = []
                loss_values_list = []
                while True:
                    try:
                        label_value, pred_value, loss_value = sess.run([labels, predictions, loss],
                                                                       feed_dict={dropout_keep_rate: 1})

                        label_values_list.extend(label_value)
                        pred_values_list.extend(pred_value)
                        loss_values_list.append(loss_value)

                    except tf.errors.OutOfRangeError:
                        kendall_dev, kendall_dev_p = kendalltau(label_values_list, pred_values_list)
                        pearson_dev, pearson_dev_p = pearsonr(label_values_list, pred_values_list)
                        loss_dev = np.mean(loss_values_list)
                        print("Validation: loss = %.4f, kendall = %.4f (%.4f), pearson = %.4f (%.4f)" % (
                            loss_dev, kendall_dev, kendall_dev_p, pearson_dev, pearson_dev_p))
                        break

                sess.run(test_init_op)
                label_values_list = []
                pred_values_list = []
                loss_values_list = []
                while True:
                    try:
                        label_value, pred_value, loss_value = sess.run([labels, predictions, loss],
                                                                       feed_dict={dropout_keep_rate: 1})

                        label_values_list.extend(label_value)
                        pred_values_list.extend(pred_value)
                        loss_values_list.append(loss_value)

                    except tf.errors.OutOfRangeError:
                        kendall_test, kendall_test_p = kendalltau(label_values_list, pred_values_list)
                        pearson_test, pearson_test_p = pearsonr(label_values_list, pred_values_list)
                        loss_dev = np.mean(loss_values_list)
                        print("Test: loss = %.4f, kendall = %.4f (%.4f), pearson = %.4f (%.4f)" % (
                            loss_dev, kendall_test, kendall_test_p, pearson_test, pearson_test_p))

                        if kendall_dev > best_score[0][0]:
                            no_improve = 0
                            path = os.path.join(FLAGS.checkpoint_dir, FLAGS.category, "model.ckpt")
                            tf.gfile.MakeDirs(path)
                            saver.save(sess, path)
                            best_score[0] = [kendall_dev, kendall_dev_p]
                            best_score[1] = [kendall_test, kendall_test_p]
                            tf.gfile.MakeDirs(FLAGS.output_dir)
                            with open(os.path.join(FLAGS.output_dir, FLAGS.category), "w") as fout:
                                for pred, label in zip(pred_values_list, label_values_list):
                                    fout.write("%f\t%f\n" % (pred, label))
                        else:
                            no_improve += 1

                        print("Best score so far:   Dev = %.4f (%f) test = %.4f(%f)" % (
                        best_score[0][0], best_score[0][1], best_score[1][0], best_score[1][1]))
                        break
