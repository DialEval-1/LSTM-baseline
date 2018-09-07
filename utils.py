import os
from urllib.request import urlretrieve
from tqdm import tqdm
import tensorflow as tf
import sys
import pickle

CN_TRAIN_PATH = "https://raw.githubusercontent.com/sakai-lab/stc3-dataset/master/data/train_data_cn.json"
EN_TRAIN_PATH = "https://raw.githubusercontent.com/sakai-lab/stc3-dataset/master/data/train_data_en.json"
EMBEDDING_PATH = "http://bytensor.com/embedding/baidu_256_500k.pkl"

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


def maybe_download(url, download_path, filename):
    if not os.path.exists(os.path.join(download_path, filename)):
        try:
            print("Downloading file {}...".format(url + filename))
            with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                local_filename, _ = urlretrieve(url, os.path.join(download_path, filename), reporthook=t.update_to)
        except AttributeError as e:
            print("An error occurred when downloading the file! Please get the dataset using a browser.")
            raise e

def load_embedding(embedding_path):
    wtoi, itow, weight = load(os.path.join(embedding_path, "baidu_256_500k.pkl"))
    return wtoi, itow, weight

def prepare_data(embedding_path, data_path):
    tf.gfile.MakeDirs(embedding_path.as_posix())
    tf.gfile.MakeDirs(str(data_path))

    maybe_download(CN_TRAIN_PATH, data_path, os.path.basename(CN_TRAIN_PATH))
    maybe_download(EN_TRAIN_PATH, data_path, os.path.basename(EN_TRAIN_PATH))
    maybe_download(EMBEDDING_PATH, data_path, os.path.basename(EMBEDDING_PATH))


    return os.path.join(data_path, os.path.basename(CN_TRAIN_PATH)), \
           os.path.join(data_path, os.path.basename(EN_TRAIN_PATH)), \
            os.path.join(data_path, os.path.basename(EMBEDDING_PATH))

def is_macos():
    return sys.platform == "darwin"


class MacOSFile(object):

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
            #             idx += batch_sizee

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

