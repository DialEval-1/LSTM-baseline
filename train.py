import tensorflow as tf
from pathlib2 import Path
from tensorflow.contrib.training import HParams

import model as model
from data import dataset
from flags import define_flags
from utils import prepare_data, load_embedding

PROJECT_DIR = Path(__file__).parent.parent


def start(customized_params=None):
    params = define_flags()
    if customized_params:
        params.__dict__update(customized_params)

    with tf.Graph().as_default():
        train_path, test_path, embedding = prepare_data(params.embedding_path, params.data_path)
        wtoi, itow, weight = load_embedding(params.embedding_path)
        train_dataset = dataset(train_path, wtoi, task=params.task, batch_size=params.batch_size)