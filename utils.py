import csv
import hashlib
import os
import pickle
import sys
import time
import zipfile
from enum import Enum, unique
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib2 import Path
from tqdm import tqdm

GLOVE_PATH = "http://nlp.stanford.edu/data/glove.840B.300d.zip"
GLOVE_MD5 = "2ffafcc9f9ae46fc8c95f32372976137"

BAIDU_PATH = "http://bytensor.com/share/embedding/baidu_256_500k.zip"
BAIDU_MD5 = "4b0e26078a5b5b9201030ed77d5045ab"


@unique
class Language(Enum):
    english = 0
    chinese = 1


@unique
class Task(Enum):
    nugget = 0
    quality = 1


@unique
class SpecialTokens(Enum):
    UNK = "###UNK###"
    PAD = "###PAD###"
    EOS = "###EOS###"
    SOS = "###SOS###"


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def maybe_download(url: str, store_path: Path, filename: str, md5: str = None):
    if not (store_path / filename).is_file():
        try:
            print("Downloading file {}...".format(url + "  " + filename))
            with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                local_filename, _ = urlretrieve(url, store_path / filename, reporthook=t.update_to)
        except AttributeError as e:
            print("An error occurred when downloading the file! Please get the dataset using a browser.")
            raise e

        if md5:
            md5_download = md5Checksum(store_path / filename)
            if not md5 == md5_download:
                store_path.joinpath(filename).unlink()
                raise ValueError("MD5 checksum error, expected %s but got %s" % (md5, md5_download))

    return store_path / filename


def load_embedding(embedding_path):
    wtoi, itow, weight = load(os.path.join(embedding_path, "baidu_256_500k.pkl"))
    return wtoi, itow, weight


def prepare_data(embedding_path, data_path, language=Language.english):
    tf.gfile.MakeDirs(str(embedding_path))
    if language == Language.english:

        train_path = data_path / "train_data_cn.json"
        test_path = data_path / "test_data_cn.json"
        embedding = GloveEmbedding(embedding_path)
    else:
        embedding = BaiduEmbedding(embedding_path)
        train_path = data_path / "train_data_en.json"
        test_path = data_path / "test_data_en.json"

    return train_path, test_path, embedding


def is_macos():
    return sys.platform == "darwin"


class MacOSFile(object):
    """
    On MacOS, pickle cannot load or dump objects which are larger than 2Gb due to an unsolved bug.
    """

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            # print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            # print("don.", flush=True)
            idx += batch_size


def dump(obj, file_path):
    with open(file_path, "wb") as f:
        if is_macos():
            f = MacOSFile(f)
        return pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(file_path):
    with open(file_path, "rb") as f:
        if is_macos():
            f = MacOSFile(f)

        return pickle.load(f)


def test_run_time(fn):
    start = time.time()
    fn()
    print("%s used %s seconds" % (str(fn), time.time() - start))


class Embedding(object):
    def __init__(self, download_url, md5, store_folder):
        self.download_url = download_url
        self.store_folder = Path(store_folder)
        self.download_file = store_folder / Path(self.download_url).name
        self.pkl_file = self.download_file.with_suffix(".pkl")

        if not self.pkl_file.is_file():
            self._download_and_unzip(md5)
        self._load_and_to_pickle()
        self._add_special_tokens()

    def _download_and_unzip(self, md5):
        if self.download_file.is_file():
            return

        if self.pkl_file.is_file():
            return

        maybe_download(self.download_url, store_path=self.store_folder, filename=self.download_file.name, md5=md5)

        if Path(self.download_url).suffix != ".zip":
            return

        with zipfile.ZipFile(self.download_file) as zf:
            # assume there is only one file in zip
            [unzipped_file_name] = zf.namelist()

            if not (self.store_folder / unzipped_file_name).is_file():
                zf.extractall(self.store_folder)

            self.download_file.unlink()
            self.download_file = self.store_folder / unzipped_file_name

    def _load_and_to_pickle(self):
        if not self.pkl_file.is_file():
            self.wtoi, self.itow, self.weight = self._parse_embedding()
            dump([self.wtoi, self.itow, self.weight], self.pkl_file)
            self.download_file.unlink()
        else:
            self.wtoi, self.itow, self.weight = load(self.pkl_file)

    def _parse_embedding(self):
        raise NotImplementedError

    def _add_special_tokens(self):
        for t in SpecialTokens:
            self.wtoi[t.value] = len(self.itow)
            self.itow.append(t.value)

        new_weight = np.zeros(shape=[len(SpecialTokens), self.weight.shape[1]])
        self.weight = np.concatenate([self.weight, new_weight], axis=0)


class GloveEmbedding(Embedding):
    def __init__(self, store_folder):
        super().__init__(GLOVE_PATH, GLOVE_MD5, store_folder)

    def _parse_embedding(self):
        df = pd.read_table(self.download_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
        itow = list(df.index)
        wtoi = {w: i for i, w in enumerate(itow)}
        weights = df.as_matrix()

        return wtoi, itow, weights


class BaiduEmbedding(Embedding):
    def __init__(self, store_folder):
        super().__init__(BAIDU_PATH, BAIDU_MD5, store_folder)

    def _parse_embedding(self):
        return load(self.store_folder / self.pkl_file)


def md5Checksum(filePath):
    BUFFER_SIZE = 8192
    with open(filePath, 'rb') as fh:
        m = hashlib.md5()
        while True:
            data = fh.read(BUFFER_SIZE)
            if not data:
                break
            m.update(data)
        return m.hexdigest()
