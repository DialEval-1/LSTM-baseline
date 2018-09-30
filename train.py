from functools import partial

import tensorflow as tf
from pathlib2 import Path
from sklearn import model_selection

import vocab
from data import process_raw_data, build_dataset_op, Task
import data
from flags import define_flags
from vocab import Language
import model
import json
import pprint

from stc3dataset.data.eval import evaluate_from_list

PROJECT_DIR = Path(__file__).parent.parent
PP = pprint.PrettyPrinter()


def flags2params(flags, customized_params=None):
    if customized_params:
        flags.__dict__.update(customized_params)

    flags.checkpoint_path = Path(flags.checkpoint_path) / flags.language / flags.task / "model"
    flags.output_path.mkdir(parents=True, exist_ok=True)

    flags.language = vocab.Language[flags.language]
    flags.task = data.Task[flags.task]
    if flags.language == Language.english:
        flags.vocab = getattr(vocab, flags.english_vocab)
    else:
        flags.vocab = getattr(vocab, flags.chinese_vocab)

    flags.optimizer = getattr(tf.train, flags.optimizer)
    flags.cell = getattr(tf.nn.rnn_cell, flags.cell)

    return flags


class TrainingHelper(object):
    def __init__(self, customized_params=None):
        # Parse parameters
        flags = define_flags()
        params = flags2params(flags, customized_params)

        self.task = params.task
        self.language = params.language

        # Loading dataset
        self.train_path, self.test_path, self.vocab = prepare_data_and_vocab(
            vocab=params.vocab,
            store_folder=params.embedding_path,
            data_path=params.data_path,
            language=params.language)

        # split training set into train and dev sets
        self.raw_train, self.raw_dev = model_selection.train_test_split(
            json.load(self.train_path.open()),
            test_size=params.dev_ratio, random_state=params.random_seed)

        self.raw_test = json.load(self.test_path.open())

        train_dataset = process_raw_data(
            self.raw_train,
            vocab=self.vocab,
            max_len=params.max_len,
            cache_path=params.cache_path,
            is_train=True,
            name="train_%s" % params.language)

        dev_dataset = process_raw_data(
            self.raw_dev,
            vocab=self.vocab,
            max_len=params.max_len,
            cache_path=params.cache_path,
            is_train=False,
            name="dev_%s" % params.language)

        test_dataset = process_raw_data(
            self.raw_test,
            vocab=self.vocab,
            max_len=params.max_len,
            cache_path=params.cache_path,
            is_train=False,
            name="test_%s" % params.language)

        pad_idx = self.vocab.pad_idx
        self.train_iterator = build_dataset_op(train_dataset, pad_idx, params.batch_size, is_train=True)
        self.train_batch = self.train_iterator.get_next()
        self.dev_iterator = build_dataset_op(dev_dataset, pad_idx, params.batch_size, is_train=False)
        self.dev_batch = self.dev_iterator.get_next()
        self.test_iterator = build_dataset_op(test_dataset, pad_idx, params.batch_size, is_train=False)
        self.test_batch = self.test_iterator.get_next()

        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)

        self.model = model.Model(self.vocab.weight, self.task, params, session=sess)

        self.checkpoint_path = params.checkpoint_path
        self.output_path = params.output_path
        if params.resume:
            self.model.load_model(self.checkpoint_path)

        self.epoch_num = params.epoch_num

    def train_epoch(self, checkpoint_path=None):
        train_loss = self.model.train_epoch(
            self.train_iterator.initializer,
            self.train_batch,
            save_path=checkpoint_path or self.checkpoint_path)
        return train_loss

    def train(self, num_epoch=None):
        for num_epoch in range(num_epoch or self.epoch_num):
            train_loss = self.train_epoch()
            print("%d Epoch, training loss = %.4f" % (num_epoch + 1, train_loss))
            print("Dev Metrics:")
            metrics = self.evaluate_on_dev()
            PP.pprint(metrics)
            print("\n\n")

    def evaluate_on_dev(self):
        predictions = self.model.predict(self.dev_iterator.initializer, self.dev_batch)
        submission = self.__predictions_to_submission_format(predictions)
        scores = evaluate_from_list(submission, self.raw_dev)
        return scores

    def predict_test(self):
        predictions = self.model.predict(self.test_iterator.initializer, self.test_batch)
        submission = self.__predictions_to_submission_format(predictions)

        output_file = trainer.output_path / ("%s_test_submission.json" % self.language.name)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        json.dump(output_file.open("w"), submission)

        return submission

    def __predictions_to_submission_format(self, predictions):
        submission = []
        for pred in predictions:
            if self.task == Task.nugget:
                submission.append(data.nugget_prediction_to_submission_format(pred))
            elif self.task == Task.quality:
                submission.append(data.quality_prediction_to_submission_format(pred))
        return submission


def prepare_data_and_vocab(vocab, store_folder, data_path, language=Language.english, tokenizer=None):
    tf.gfile.MakeDirs(str(store_folder))
    if language == Language.chinese:
        train_path = data_path / "train_data_cn.json"
        test_path = data_path / "test_data_cn.json"
    else:
        train_path = data_path / "train_data_en.json"
        test_path = data_path / "test_data_en.json"

    vocab = vocab(store_folder, tokenizer=tokenizer)
    return train_path, test_path, vocab


if __name__ == '__main__':
    trainer = TrainingHelper()
    trainer.train(num_epoch=15)
    test_prediction = trainer.predict_test()
