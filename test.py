import unittest
from unittest import TestCase
from pathlib2 import Path
from utils import GloveEmbedding, BaiduEmbedding
import tensorflow as tf

DATA_PATH = Path(__file__).parent / "data"
EMBEDDING_PATH = DATA_PATH / "embedding"

tf.gfile.MakeDirs(str(EMBEDDING_PATH.resolve()))

class TestEmbedding(TestCase):

    def test_GloveEmbedding(self):
        glove = GloveEmbedding(EMBEDDING_PATH)
        assert len(glove.wtoi) > 1000
        assert glove.weight.shape[1] == 300
    def test_BaiduEmbedding(self):
        baidu = BaiduEmbedding(EMBEDDING_PATH)
        assert len(baidu.wtoi) > 1000



if __name__ == "__main__":
    glove = GloveEmbedding(EMBEDDING_PATH)
    baidu = BaiduEmbedding(EMBEDDING_PATH)