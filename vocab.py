import csv
import logging
import zipfile
from enum import Enum, unique

import jieba
import nltk

import numpy as np
import pandas as pd
import spacy
from pathlib2 import Path

from utils import maybe_download, dump, load

GLOVE_PATH = "http://nlp.stanford.edu/data/glove.840B.300d.zip"
GLOVE_MD5 = "2ffafcc9f9ae46fc8c95f32372976137"

BAIDU_PATH = "http://bytensor.com/share/embedding/baidu_256_500k.zip"
BAIDU_MD5 = "4b0e26078a5b5b9201030ed77d5045ab"

GLOVE_6B_PATH = "http://nlp.stanford.edu/data/glove.6B.zip"


@unique
class Language(Enum):
    english = 0
    chinese = 1


@unique
class SpecialTokens(Enum):
    UNK = "###UNK###"
    PAD = "###PAD###"
    EOS = "###EOS###"
    SOS = "###SOS###"


class Vocab(object):
    def __init__(self, store_folder, download_url, md5=None, file_to_extract=None, language=Language.english,
                 tokenizer=None, cased=True):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Loading embedding from: %s" % download_url)

        self.download_url = download_url
        self.store_folder = Path(store_folder)
        self.download_file = store_folder / Path(self.download_url).name

        if file_to_extract is None:
            self.pkl_file = self.download_file.with_suffix(".pkl")
        else:
            self.pkl_file = (self.store_folder / file_to_extract).with_suffix(".pkl")

        self._download_and_unzip(md5, file_to_extract)
        self._load_and_to_pickle()
        self._add_special_tokens()

        self.unk_idx = self._wtoi[SpecialTokens.UNK.value]
        self.pad_idx = self._wtoi[SpecialTokens.PAD.value]
        self.eos_idx = self._wtoi[SpecialTokens.EOS.value]
        self.sos_idx = self._wtoi[SpecialTokens.SOS.value]

        self.language = language
        self.cased = cased

        if tokenizer is None:
            if self.language is Language.english:
                self.tokenizer = get_tokenizer("spacy:en")

            elif self.language is Language.chinese:
                self.tokenizer = get_tokenizer("jieba:lcut_for_search")
            else:
                raise NotImplementedError

        else:
            self.tokenizer = get_tokenizer(tokenizer)

        self.logger.info("Set tokenizer to %s" % self.tokenizer.__name__)
        self.logger.info("Loaded vocab (embedding) from: %s" % download_url)

    def word_to_index(self, word):
        """
        :param word:
        :return: word index. if word is out-of-vocabulary, unk_idx will be returned
        """
        if not self.cased:
            word = word.lower()

        return self._wtoi.get(word, self.unk_idx)

    def index_to_word(self, index):
        return self._itow[index]

    def _download_and_unzip(self, md5=None, file_to_extract=None):
        if self.pkl_file.is_file():
            self.logger.info("Found binary format embedding %s" % self.pkl_file)
            return

        maybe_download(self.download_url, store_path=self.store_folder, filename=self.download_file.name, md5=md5)

        if Path(self.download_url).suffix != ".zip":
            return

        with zipfile.ZipFile(self.download_file) as zf:
            # assume there is only one file in zip

            if file_to_extract is None:
                [unzipped_file_name] = zf.namelist()
            else:
                unzipped_file_name = file_to_extract

            if not (self.store_folder / unzipped_file_name).is_file():
                self.logger.info("Unzipping the embedding file %s" % self.download_file)
                zf.extract(member=unzipped_file_name, path=self.store_folder)

            self.logger.info("Delete the zip file %s" % self.download_file)
            self.download_file.unlink()
            self.download_file = self.store_folder / unzipped_file_name

    def _load_and_to_pickle(self):
        if not self.pkl_file.is_file():
            try:
                self.logger.info("Loading the downloaded embedding from %s" % self.download_file)
                self._wtoi, self._itow, self.weight = self._parse_embedding()

                self.logger.info("Saving the embedding as a binary file to %s" % self.pkl_file)
                dump([self._wtoi, self._itow, self.weight], self.pkl_file)
                self.download_file.unlink()
            except MemoryError as e:
                self.logger.error("Current RAM (not GPU Memory) is not enough"
                                  " to load the word embedding file %s. " % self.download_file)
                raise e

        else:
            self._wtoi, self._itow, self.weight = load(self.pkl_file)

    def _parse_embedding(self):
        raise NotImplementedError

    def _add_special_tokens(self):
        for t in SpecialTokens:
            self._wtoi[t.value] = len(self._itow)
            self._itow.append(t.value)
        # noinspection PyTypeChecker
        new_weight = np.zeros(shape=[len(SpecialTokens), self.weight.shape[1]])
        self.weight = np.concatenate([self.weight, new_weight], axis=0)


class Glove840B(Vocab):
    def __init__(self, store_folder, download_url=GLOVE_PATH, md5=GLOVE_MD5, tokenizer=None):
        super().__init__(store_folder, download_url, md5, language=Language.english, tokenizer=tokenizer)

    def _parse_embedding(self):
        df = pd.read_table(self.download_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
        itow = list(df.index)
        wtoi = {w: i for i, w in enumerate(itow)}
        weights = df.as_matrix()

        return wtoi, itow, weights


class Glove6B(Vocab):
    def __init__(self, store_folder, download_url=GLOVE_6B_PATH, tokenizer=None):
        super().__init__(store_folder, download_url, md5=None, file_to_extract="glove.6B.50d.txt",
                         language=Language.english, tokenizer=tokenizer, cased=False)

    def _parse_embedding(self):
        df = pd.read_table(self.download_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
        itow = list(df.index)
        wtoi = {w: i for i, w in enumerate(itow)}
        weights = df.as_matrix()

        return wtoi, itow, weights


class Baidu(Vocab):
    def __init__(self, store_folder, download_url=BAIDU_PATH, md5=BAIDU_MD5, tokenizer=None):
        super().__init__(store_folder, download_url, md5, language=Language.chinese, tokenizer=tokenizer)

    def _parse_embedding(self):
        return load(self.store_folder / self.pkl_file)


class SpacyTokenizer(object):
    def __init__(self, lang="en"):
        self.nlp = spacy.load(lang)

    def __call__(self, seq):
        return [w.text for w in self.nlp(seq)]

    __name__ = "Spacy"


def get_tokenizer(tokenizer="spacy:en"):
    if tokenizer == "nltk":
        return nltk.word_tokenize

    elif tokenizer.startswith("spacy"):
        elements = tokenizer.split(":")
        if len(elements) == 1:
            lang = "en"
        elif len(elements) == 2:
            lang = elements[1]
        else:
            raise ValueError("tokenizer name is not appropriate")
        return SpacyTokenizer(lang)

    elif tokenizer.startswith("jieba"):

        elements = tokenizer.split(":")
        if len(elements) == 1:
            func = "lcut_for_search"
        elif len(elements) == 2:
            func = elements[1]
        else:
            raise ValueError("tokenizer name is not appropriate")

        return getattr(jieba, func)
    else:
        raise ValueError("Invalid tokenizing method %s" % tokenizer)
