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

    flags = parser.parse_args()

    return flags


def __define_base(parser):
    parser.add_argument("--task", type=str, default="nugget")
    parser.add_argument("--language", type=str, default="english")
    parser.add_argument("--embedding-dir", type=str, default=PROJECT_DIR / "data" / "embedding")
    parser.add_argument("--cache-dir", type=str, default=PROJECT_DIR / "data" / "cache")
    parser.add_argument("--data-dir", type=str, default=PROJECT_DIR / "dataset")
    parser.add_argument("--output-dir", type=str, default=PROJECT_DIR / "output")
    parser.add_argument("--log-dir", type=str, default=PROJECT_DIR / "log")
    parser.add_argument("--checkpoint-dir", type=str, default=PROJECT_DIR / "checkpoint")
    parser.add_argument("--best-model-dir", type=str, default=PROJECT_DIR / "best_model")
    parser.add_argument("--infer-test", type=bool, default=False)
    parser.add_argument("--resume-dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default="baseline")

def __define_training(parser):
    parser.add_argument("--num-epoch", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=.3)
    parser.add_argument("--update-embedding", type=bool, default=False)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dev-ratio", type=float, default=.2)
    parser.add_argument("--random-seed", type=int, default=2018)
    parser.add_argument("--optimizer", type=str, default="AdamOptimizer")
    parser.add_argument("--quality-primary-metric", type=str, default="nmd")  # For model selection
    parser.add_argument("--nugget-primary-metric", type=str, default="rnss")
    parser.add_argument("--trace", action="store_true", default=False)

def __define_model(parser):
    parser.add_argument("--english-vocab", type=str, default="Glove840B")
    parser.add_argument("--chinese-vocab", type=str, default="Baidu")
    parser.add_argument("--hidden-size", type=int, default=150)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--cell", type=str, default="LSTMCell")


def __define_preprocessing(parser):
    parser.add_argument("--max-len", type=int, default=100)
    parser.add_argument("--tokenizer", type=str, default=None)