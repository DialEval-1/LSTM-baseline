import json
from functools import partial

import jieba
import tensorflow as tf
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from flags import define_flags
from utils import Task
from utils import load_embedding
from utils import prepare_data

QUALITY_MEASURES = ("A", "E", "S")
C_NUGGET_TYPES = ('CNUG0', 'CNUG', 'CNUG*', 'CNaN')
H_NUGGET_TYPES = ('HNUG', 'HNUG*', 'HNaN')
QUALITY_SCALES = ('2', '1', '0', '-1', '-2')


def get_nugget2ix():
    custom_nugget_2_ix = {
        "PAD": 0,
        "CNaN": 1,
        "CNUG": 2,
        "CNUG*": 3,
        "CNUG0": 4,
    }

    helpdesk_nugget_2_ix = {
        "PAD": 0,
        "HNaN": 1,
        "HNUG": 2,
        "HNUG*": 3
    }

    nugget_2_idx = custom_nugget_2_ix.copy()
    nugget_2_idx.update(helpdesk_nugget_2_ix)
    return custom_nugget_2_ix, helpdesk_nugget_2_ix, nugget_2_idx


def parse_nugget_label(annotations):
    pass


def parse_quality_label(annotations):
    pass



def get_special_idx(wtoi):
    if "##UNK##" in wtoi:
        unk_id = wtoi["##UNK##"]
    elif "<unk>" in wtoi:
        unk_id = wtoi["<unk>"]
    else:
        raise KeyError("Cannot find unk in vocabulary")

    return unk_id


def read_json_generator(data_path,
                        wtoi,
                        task=Task.nugget,
                        tokenizer=jieba.cut_for_search,
                        max_len=100):
    data = json.load(open(data_path, encoding="utf-8"))

    unk_id = get_special_idx(wtoi)

    for dialogue in data:
        tokenized_turns = []
        senders = []
        for turn in dialogue["dialog"]["turns"]:
            sender = 1 if turn["sender"].startswith("c") else 0
            senders.append(sender)
            text = " ".join(turn["utterances"])
            tokenized_text = [wtoi.get(token, unk_id) for token in tokenizer(text[:max_len])]
            tokenized_turns.append(tokenized_text)


        if task == Task.nugget:
            parse_fn = parse_nugget_label
        elif task == Task.quality:
            parse_fn = parse_nugget_label
        else:
            raise ValueError("Incorrect task %s" % task)

        label = parse_fn(dialogue["annotations"])
        utterance_lengths = [len(u) for u in tokenized_turns]
        dialogue_length = len(tokenized_turns)
        padded_utterance = pad_sequences(tokenized_turns,padding="post", truncating="post")

        yield (padded_utterance,
               senders,
               utterance_lengths,
               dialogue_length,
               label)



def dataset(data_path, wtoi,task=Task.nugget, batch_size=32, shuffle=True):
    data_gen = partial(read_json_generator,
                       wtoi=wtoi,
                       data_path=data_path,
                       task=task)


    dataset = tf.data.Dataset.from_generator(data_gen,
                                             output_types=(tf.int32, tf.bool, tf.int32, tf.int32, tf.float32))

    if shuffle:
        dataset = dataset.shuffle(5000)

    if task == Task.nugget:
        label_shape = tf.TensorShape([None])
    elif  task == Task.quality :
        label_shape = tf.TensorShape([])
    else:
        raise ValueError("Incorrect task %s" % task)


    dataset = dataset.padded_batch(batch_size=batch_size,
                                   padded_shapes=(tf.TensorShape([None, None]),
                                                  tf.TensorShape([None]),
                                                  tf.TensorShape([None]),
                                                  tf.TensorShape([]),
                                                  label_shape),
                                   padding_values=(0, False, 0, 0, 0.))

    return dataset


if __name__ == "__main__":
    params = define_flags()

    train_path, dev_path, test_path = prepare_data(params.embedding_path, params.data_path)
    wtoi, itow, weight = load_embedding(params.embedding_path)
    train_dataset = dataset(train_path, wtoi, task=Task.nugget, batch_size=params.batch_size)
    dev_dataset = dataset(dev_path, wtoi, task=Task.nugget, batch_size=params.batch_size * 3)
    test_dataset = dataset(test_path, wtoi, task=Task.nugget, batch_size=params.batch_size * 3)

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