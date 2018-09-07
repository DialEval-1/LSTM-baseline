import json
from collections import Counter
import random
import numpy as np
from functools import partial
import jieba
import tensorflow as tf

from baseline.utils import prepare_data, load_embedding


def get_nugget2ix():
    custom_nugget_2_ix = {
        "PAD": 0,
        "NaN": 1,
        "CNUG0": 2,
        "CNUG": 3,
        "CNUG*": 4
    }

    helpdesk_nugget_2_ix = {
        "PAD": 0,
        "NaN": 1,
        "HNUG": 2,
        "HNUG*": 3
    }

    nuuget_2_idx = custom_nugget_2_ix.copy()
    nuuget_2_idx.update(helpdesk_nugget_2_ix)
    return custom_nugget_2_ix, helpdesk_nugget_2_ix, nuuget_2_idx


def find_majority(seqs):
    c = Counter(seqs)
    most_common, freq = c.most_common()[0]
    if freq > len(seqs) // 2 or freq > 1:
        return most_common
    return random.choice(seqs)

def stupid_tokenizer(seq):
    return seq.split()

def read_dch_json_gen(data_path,
                      wtoi,
                      merge_subjective_scores=np.mean,
                      label_category="CS",
                      tokenizer=jieba.cut_for_search,
                      max_len=100):
    data = json.load(open(data_path, encoding="utf-8"))["data"]

    if "##UNK##" in wtoi:
        unk_id = wtoi["##UNK##"]
    elif "<unk>" in wtoi:
        unk_id = wtoi["<unk>"]
    else:
        raise KeyError("Cannot find unk in vocabulary")

    for example in data:
        utterances_per_example = []
        senders_per_example = []
        # write utterance and sender
        for utterance in example["dialog"]["content"]:
            sender = 1 if utterance["sender"].startswith("c") else 0
            senders_per_example.append(sender)
            utterances_per_example.append(
                [wtoi.get(token, unk_id) for token in tokenizer(utterance["utterance"][:max_len])])

        labels = {}
        nuggets = []
        for annotation in example["annotation"]:
            # fetch subjective score
            for sub, score in annotation["subjective"].items():
                labels.setdefault(sub, [])

                if sub != "CS":
                    score = score - 1
                else:
                    score = score

                labels[sub].append(score)
            nuggets.append(annotation["nugget"])

        # merge subjective scores from different annotators
        for sub, scores in labels.items():
            labels[sub] = merge_subjective_scores(scores) if merge_subjective_scores else scores
        _, _, nugget_2_ix = get_nugget2ix()
        # merge nuggets
        nuggets = [nugget_2_ix[find_majority(nuggets)] for nuggets in zip(*nuggets)]
        labels["nugget"] = nuggets

        utterance_lengths = [len(u) for u in utterances_per_example]
        dialogue_length = len(utterances_per_example)

        padded_utterance = tf.keras.preprocessing.sequence.pad_sequences(utterances_per_example,
                                                                         padding="post", truncating="post")


        if label_category in labels:
            yield (padded_utterance,
                   senders_per_example,
                   utterance_lengths,
                   dialogue_length,
                   labels[label_category])
        else:
            raise ValueError("%s is not a valid label category" % label_category)


def dataset(data_path, wtoi, label_category="CS", batch_size=32, shuffle=True):
    data_gen = partial(read_dch_json_gen,
                       wtoi=wtoi,
                       data_path=data_path,
                       label_category=label_category)

    label_dtype = tf.int32 if label_category == "nugget" else tf.float32
    dataset = tf.data.Dataset.from_generator(data_gen,
                                             output_types=(tf.int32, tf.bool, tf.int32, tf.int32, label_dtype))

    if shuffle:
        dataset = dataset.shuffle(5000)

    label_shape = tf.TensorShape([None]) if label_category == "nugget" else tf.TensorShape([])
    dataset = dataset.padded_batch(batch_size=batch_size,
                                   padded_shapes=(tf.TensorShape([None, None]),
                                                  tf.TensorShape([None]),
                                                  tf.TensorShape([None]),
                                                  tf.TensorShape([]),
                                                  label_shape),
                                   padding_values=(0, False, 0, 0, 0.))

    return dataset


if __name__ == "__main__":
    from baseline.main import FLAGS

    train_path, dev_path, test_path = prepare_data(FLAGS.embedding_path, FLAGS.dch_path)
    wtoi, itow, weight = load_embedding(FLAGS.embedding_path)
    train_dataset = dataset(train_path, wtoi, label_category="CS", batch_size=FLAGS.batch_size)
    dev_dataset = dataset(dev_path, wtoi, label_category="CS", batch_size=FLAGS.batch_size * 3)
    test_dataset = dataset(test_path, wtoi, label_category="CS", batch_size=FLAGS.batch_size * 3)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)

    training_init_op = iterator.make_initializer(train_dataset)
    dev_init_op = iterator.make_initializer(dev_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    with tf.device("cpu:0"):
        batch = iterator.get_next()

    with tf.Session() as sess:
        sess.run(training_init_op)
        b = sess.run(batch)
        for i in range(len(b)):
            print(b[i].shape)
