from argparse import ArgumentParser
from pathlib2 import Path
import vocab

PROJECT_DIR = Path(__file__).parent.resolve()


def define_flags():
    parser = ArgumentParser()

    __define_base(parser)
    __define_model(parser)
    __define_training(parser)
    __define_preprocessing(parser)

    flags, _ = parser.parse_known_args()

    return flags


def __define_base(parser):
    parser.add_argument("--task", type=str, default="nugget")
    parser.add_argument("--language", type=str, default="chinese")
    parser.add_argument("--embedding-path", type=str, default=PROJECT_DIR / "data" / "embedding")
    parser.add_argument("--cache-path", type=str, default=PROJECT_DIR / "data" / "cache")
    parser.add_argument("--data-path", type=str, default=PROJECT_DIR / "stc3dataset" / "data")
    parser.add_argument("--output-path", type=str, default=PROJECT_DIR / "output")
    parser.add_argument("--checkpoint-path", type=str, default=PROJECT_DIR / "checkpoint")
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--resume", action="store_true", default=False)

def __define_training(parser):
    parser.add_argument("--epoch-num", type=int, default=50)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=.3)
    parser.add_argument("--update-embedding", type=bool, default=False)
    parser.add_argument("--grad-clip", type=float, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--k-folder", type=int, default=4)
    parser.add_argument("--dev-ratio", type=float, default=.2)
    parser.add_argument("--random-seed", type=int, default=2018)
    parser.add_argument("--optimizer", type=str, default="AdamOptimizer")
    parser.add_argument("--quality-primary-metric", type=str, default="nmd")  # For model selection
    parser.add_argument("--nugget-primary-metric", type=str, default="rnss")

def __define_model(parser):
    parser.add_argument("--english-vocab", type=str, default="Glove6B")
    parser.add_argument("--chinese-vocab", type=str, default="Baidu")
    parser.add_argument("--hidden-size", type=int, default=150)
    parser.add_argument("--num-layer", type=int, default=3)
    parser.add_argument("--cell", type=str, default="LSTMCell")


def __define_preprocessing(parser):
    parser.add_argument("--max-len", type=int, default=100)
    parser.add_argument("--tokenizer", type=str, default=None)
