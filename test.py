import logging
import unittest
from unittest import TestCase

import jieba
import tensorflow as tf
from pathlib2 import Path
from vocab import get_tokenizer
from data import process_raw_data
from vocab import Glove840B, Baidu, Glove6B
import train

DATA_PATH = Path(__file__).parent / "stc3-dataset" / "data"
EMBEDDING_PATH = Path(__file__).parent / "data" / "embedding"
CACHE_PATH = Path(__file__).parent / "data" / "cache"

tf.gfile.MakeDirs(str(EMBEDDING_PATH.resolve()))
tf.gfile.MakeDirs(str(CACHE_PATH.resolve()))

logging.basicConfig(level=logging.DEBUG)
jieba.setLogLevel(logging.INFO)

en_tokenizer = get_tokenizer("spacy")
cn_tokenizer = get_tokenizer("jieba")



class TestDataset(TestCase):
    @classmethod
    def setUpClass(cls):
        """ get_some_resource() is slow, to avoid calling it for each test use setUpClass()
            and store the result as class variable
        """
        super().setUpClass()

        cls.glove = Glove6B(EMBEDDING_PATH)
        cls.baidu = Baidu(EMBEDDING_PATH)

    def test_GloveEmbedding(self):
        self.assertTrue(len(self.glove._wtoi) > 1000)
        self.assertEqual(self.glove.weight.shape[1], 50)

    def test_BaiduEmbedding(self):
        self.assertGreater(len(self.baidu._wtoi), 1000)
        self.assertEqual(self.baidu.weight.shape[1], 256)

    def test_load_cn_raw_dataset(self):
        dataset = process_raw_data(DATA_PATH / "train_data_cn.json", self.baidu,
                                   cache_dir=CACHE_PATH)
        self.assertEqual(len(dataset), 3700)

        expected_senders = [(i + 1) % 2 for i in range(500)]
        for example in dataset:
            (padded_utterance,
             senders,
             utterance_lengths,
             dialogue_length,
             customer_nugget_label,
             helpdesk_nugget_label,
             quality_label) = example

            self.assertEqual(senders, expected_senders[:len(senders)])
            self.assertEqual(len(customer_nugget_label) + len(helpdesk_nugget_label), dialogue_length)
            self.assertEqual(len(padded_utterance), dialogue_length)

        print(dataset[15])

    def test_load_en_raw_dataset(self):
        dataset = process_raw_data(DATA_PATH / "train_data_en.json", self.glove,
                                   cache_dir=CACHE_PATH)
        self.assertEqual(len(dataset), 1672)

        expected_senders = [(i + 1) % 2 for i in range(500)]
        for example in dataset:
            (padded_utterance,
             senders,
             utterance_lengths,
             dialogue_length,
             customer_nugget_label,
             helpdesk_nugget_label,
             quality_label) = example

            self.assertEqual(senders, expected_senders[:len(senders)])
            self.assertEqual(len(customer_nugget_label) + len(helpdesk_nugget_label), dialogue_length)
            self.assertEqual(len(padded_utterance), dialogue_length)

        print(dataset[15])


class TestTrain(TestCase):
    @classmethod
    def setUpClass(cls):
        """ get_some_resource() is slow, to avoid calling it for each test use setUpClass()
            and store the result as class variable
        """
        super().setUpClass()

        cls.trainer = train.TrainingHelper({"language": "chinese"})

    def test_raw_data(self):
        def print_utterance(example):
            utterances = example[0]
            for u in utterances:
                s = [self.trainer.vocab.index_to_word(i) for i in u]
                print(s)

        raw_data = self.trainer.raw_train




if __name__ == "__main__":
    unittest.main()
