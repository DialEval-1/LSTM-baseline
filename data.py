import itertools
import json
import logging
from collections import Counter
from enum import unique, Enum

import numpy as np
import tensorflow as tf
from pathlib2 import Path
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from vocab import SpecialTokens
from vocab import Vocab
import utils

QUALITY_MEASURES = ("A", "E", "S")
CUSTOMER_NUGGET_TYPES_WITH_PAD = ('PAD', 'CNUG0', 'CNUG', 'CNUG*', 'CNaN')
HELPDESK_NUGGET_TYPES_WITH_PAD = ('PAD', 'HNUG', 'HNUG*', 'HNaN')
QUALITY_SCALES = ('2', '1', '0', '-1', '-2')

C_NUGGET_TYPES_TO_INDEX = {
    "PAD": 0,
    "CNaN": 1,
    "CNUG": 2,
    "CNUG*": 3,
    "CNUG0": 4,

}

H_NUGGET_TYPES_TO_INDEX = {
    "PAD": 0,
    "HNaN": 1,
    "HNUG": 2,
    "HNUG*": 3,
}
SHUFFLE_BUFFER_SIZE = 5000


def parse_labels(annotations, senders):
    customer_nuggets, helpdesk_nuggets = _parse_nugget_label(annotations, senders)
    return customer_nuggets, helpdesk_nuggets, _parse_quality_label(annotations)


def _parse_nugget_label(annotations, senders):
    customer_labels = []
    helpdesk_labels = []
    for i, nuggets in enumerate(zip(*(anno["nugget"] for anno in annotations))):
        nugget_types = CUSTOMER_NUGGET_TYPES_WITH_PAD if senders[i] else HELPDESK_NUGGET_TYPES_WITH_PAD
        count = Counter(nuggets)
        distribution = []
        for nugget_type in nugget_types:
            distribution.append(count.get(nugget_type, 0))

        distribution = np.array(distribution, dtype=np.float32)
        assert distribution.sum() == sum(count.values()), distribution
        distribution = distribution / distribution.sum()

        if senders[i]:
            customer_labels.append(distribution)
        else:
            helpdesk_labels.append(distribution)

    return customer_labels, helpdesk_labels


def _parse_quality_label(annotations):
    label = {}

    for anno in annotations:
        for measure in QUALITY_MEASURES:
            label.setdefault(measure, [])
            label[measure].append(anno["quality"][measure])

    for measure, values in label.items():
        distribution = []
        count = Counter(values)
        for scale in QUALITY_SCALES:
            distribution.append(count.get(int(scale), 0))

        distribution = np.array(distribution, dtype=np.float32)
        assert distribution.sum() == sum(count.values()), distribution
        distribution /= distribution.sum()

        label[measure] = distribution

    label = [label[measure] for measure in QUALITY_MEASURES]

    return np.stack(label)


def get_special_idx(embedding: Vocab):
    unk_idx = embedding._wtoi[SpecialTokens.UNK.value]
    pad_idx = embedding._wtoi[SpecialTokens.PAD.value]
    sos_idx = embedding._wtoi[SpecialTokens.SOS.value]
    eos_idx = embedding._wtoi[SpecialTokens.EOS.value]

    return unk_idx, pad_idx, sos_idx, eos_idx


def process_raw_data(raw_data,
                     vocab,
                     max_len=300,
                     cache_dir=None,
                     is_train=True,
                     name="",
                     ):
    logger = logging.getLogger(__name__)

    def data_pkl_name():
        pkl_name = "_".join([name, vocab.__class__.__name__, vocab.tokenizer.__name__, str(max_len)]) + ".pkl"
        return pkl_name

    def data_gen():
        for dialogue in raw_data:
            tokenized_turns = []
            senders = []
            for turn in dialogue["turns"]:
                sender = 1 if turn["sender"].startswith("c") else 0
                senders.append(sender)
                text = " ".join(turn["utterances"])
                tokenized_text = [vocab.sos_idx]

                for token in vocab.tokenizer(text)[:max_len]:
                    tokenized_text.append(vocab.word_to_index(token))
                tokenized_text.append(vocab.eos_idx)
                tokenized_turns.append(tokenized_text)

            turn_lengths = [len(u) for u in tokenized_turns]
            dialogue_length = len(tokenized_turns)
            padded_turns = pad_sequences(tokenized_turns, padding="post",
                                         truncating="post", value=vocab.pad_idx)

            if is_train:
                customer_nugget_label, helpdesk_nugget_label, quality_label = \
                    parse_labels(dialogue["annotations"], senders)
                yield (dialogue["id"],
                       padded_turns,
                       senders,
                       turn_lengths,
                       dialogue_length,
                       customer_nugget_label,
                       helpdesk_nugget_label,
                       quality_label)

            else:
                yield (dialogue["id"],
                       padded_turns,
                       senders,
                       turn_lengths,
                       dialogue_length)

    if cache_dir:
        tf.gfile.MakeDirs(str(cache_dir))
        pkl_data = cache_dir / data_pkl_name()

        if not pkl_data.is_file():
            data = [x for x in data_gen()]

            if not is_train:
                data.sort(key=lambda x: x[4], reverse=True)

            logger.debug("Cache not found, loaded data from scratch")
            utils.dump(data, pkl_data)
            logger.debug("dumped binary dataset into %s " % pkl_data)
        else:
            data = utils.load(pkl_data)
            logger.debug("Loaded binary dataset from %s " % pkl_data)
    else:
        logger.debug("Cache_dir was not set, processed data from scratch")
        data = [x for x in data_gen()]
    logger.debug("Dataset size: %d, type: %s" % (len(data), "training" if is_train else "inference"))
    return data


@unique
class Task(Enum):
    nugget = 0
    quality = 1


def build_dataset_op(data, pad_idx, batch_size=32, is_train=True):
    if is_train:
        dataset = tf.data.Dataset.from_generator(
            lambda: (x for x in data),
            output_types=(tf.string, tf.int32, tf.bool, tf.int32, tf.int32, tf.float32, tf.float32, tf.float32))

        dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE)

        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=(tf.TensorShape([]),
                           tf.TensorShape([None, None]),
                           tf.TensorShape([None]),
                           tf.TensorShape([None]),
                           tf.TensorShape([]),
                           tf.TensorShape([None, len(CUSTOMER_NUGGET_TYPES_WITH_PAD)]),  # customer nugget label
                           tf.TensorShape([None, len(HELPDESK_NUGGET_TYPES_WITH_PAD)]),  # helpdesk nugget label
                           tf.TensorShape([len(QUALITY_MEASURES), len(QUALITY_SCALES)])  # quality label
                           ),

            padding_values=("",
                            pad_idx,
                            False,
                            0,
                            0,
                            1. / len(CUSTOMER_NUGGET_TYPES_WITH_PAD),
                            1. / len(HELPDESK_NUGGET_TYPES_WITH_PAD),
                            0.))

    else:
        dataset = tf.data.Dataset.from_generator(
            lambda: (x for x in data),
            output_types=(tf.string, tf.int32, tf.bool, tf.int32, tf.int32))

        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=(tf.TensorShape([]),
                           tf.TensorShape([None, None]),
                           tf.TensorShape([None]),
                           tf.TensorShape([None]),
                           tf.TensorShape([]),
                           ),

            padding_values=("",
                            pad_idx,
                            False,
                            0,
                            0))

    return dataset.make_initializable_iterator()


def customer_nugget_pred_to_dict(distribution):
    result = {}
    for nugget_type, prob in zip(CUSTOMER_NUGGET_TYPES_WITH_PAD, distribution):
        if nugget_type == "PAD":
            continue
        result[nugget_type] = float(prob)
    return result


def helpdesk_nugget_pred_to_dict(distribution):
    result = {}
    for nugget_type, prob in zip(HELPDESK_NUGGET_TYPES_WITH_PAD, distribution):
        if nugget_type == "PAD":
            continue
        result[nugget_type] = float(prob)
    return result


def nugget_prediction_to_submission_format(prediction):
    dialog_id, (c_nugget_pred_list, h_nugget_pred_list), dialog_length = prediction
    nugget_list = []
    for c_nugget, h_nugget in itertools.zip_longest(c_nugget_pred_list, h_nugget_pred_list):
        if c_nugget is not None:
            nugget_list.append(customer_nugget_pred_to_dict(c_nugget))
        else:
            break
        if h_nugget is not None:
            nugget_list.append(helpdesk_nugget_pred_to_dict(h_nugget))
        else:
            break

    nugget_list = nugget_list[:dialog_length]
    assert len(nugget_list) == dialog_length

    submission_format  = {
        "id": dialog_id.decode("utf-8"),
        "nugget": nugget_list
    }

    return submission_format


def quality_prediction_to_submission_format(prediction):
    dialog_id, (quality_pred), dialog_length = prediction
    result = {}
    for measure, quality_pred in zip(QUALITY_MEASURES, quality_pred):
        result[str(measure)] = {}
        for scale, prob in zip(QUALITY_SCALES, quality_pred):
            result[str(measure)][str(scale)] = float(prob)

    submission_format = {
        "id": dialog_id.decode("utf-8"),
        "quality": result
    }

    return submission_format