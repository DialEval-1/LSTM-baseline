import hashlib
import logging
import pickle
import sys
from urllib.request import urlretrieve

from pathlib2 import Path
from tqdm import tqdm

BUFFER_SIZE = 8192
FILE_SIZE_LIMIT = 1 << 31  # Applied to MacOS filesystem


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
    logger = logging.getLogger(__name__)

    if not (store_path / filename).is_file():
        try:
            logger.info("Downloading file {}...".format(url + "  " + filename))
            with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                local_filename, _ = urlretrieve(url, store_path / filename, reporthook=t.update_to)
        except AttributeError as e:
            logger.error("An error occurred when downloading the file! Please get the dataset using a browser.")
            raise e

        if md5:
            md5_download = file_md5(store_path / filename)
            if not md5 == md5_download:
                store_path.joinpath(filename).unlink()
                raise ValueError("MD5 checksum error, expected %s but got %s" % (md5, md5_download))

    return store_path / filename


def file_md5(file_path):
    with open(file_path, 'rb') as fh:
        m = hashlib.md5()
        while True:
            data = fh.read(BUFFER_SIZE)
            if not data:
                break
            m.update(data)
        return m.hexdigest()


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
        if n >= (FILE_SIZE_LIMIT):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, FILE_SIZE_LIMIT - 1)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, FILE_SIZE_LIMIT - 1)
            self.f.write(buffer[idx:idx + batch_size])
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
