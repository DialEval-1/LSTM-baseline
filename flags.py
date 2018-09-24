from argparse import ArgumentParser
from pathlib2 import Path

PROJECT_DIR = Path(__file__).parent.resolve()


def define_flags():
    parser = ArgumentParser()

    __define_base(parser)
    __define_model(parser)
    __define_training(parser)

    flags, _ = parser.parse_known_args()

    return flags


def __define_base(parser):
    parser.add_argument("--task", type=str, default="nugget")
    parser.add_argument("--language", type=str, default="english")
    parser.add_argument("--embedding-path", type=str, default=PROJECT_DIR / "data" / "embedding")
    parser.add_argument("--data-path", type=str, default=PROJECT_DIR / "data")
    parser.add_argument("--output-dir", type=str, default=PROJECT_DIR / "output")
    parser.add_argument("--checkpoint-dir", type=str, default=PROJECT_DIR / "checkpoint")
    parser.add_argument("--test", type=bool, default=False)


def __define_training(parser):
    parser.add_argument("--epoch-num", type=int, default=50)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=.3)
    parser.add_argument("--update-embedding", type=bool, default=False)
    parser.add_argument("--grad-clip", type=float, default=5)


def __define_model(parser):
    parser.add_argument("--hidden-size", type=int, default=150)
    parser.add_argument("--num-layer", type=int, default=3)
