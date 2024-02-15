from pydantic import BaseModel
from firecore_config import add_arguments
from argparse import ArgumentParser
from pathlib import Path
from typing import List


def test_int():
    class Options(BaseModel):
        max_epochs: int = 100

    parser = ArgumentParser()
    add_arguments(parser, Options())
    ns = parser.parse_args([])
    assert ns.max_epochs == 100
    ns = parser.parse_args(["--max-epochs", "50"])
    assert ns.max_epochs == 50


def test_float():
    class Options(BaseModel):
        lr: float = 1.0

    parser = ArgumentParser()
    add_arguments(parser, Options())
    ns = parser.parse_args([])
    assert ns.lr == 1.0
    ns = parser.parse_args(["--lr", "50"])
    assert ns.lr == 50.0
    assert isinstance(ns.lr, float)


def test_path():
    class Options(BaseModel):
        input_dir: Path = Path("/tmp")

    parser = ArgumentParser()
    add_arguments(parser, Options())
    ns = parser.parse_args([])
    assert ns.input_dir == Path("/tmp")
    ns = parser.parse_args(["--input-dir", "/root"])
    assert ns.input_dir == Path("/root")
    assert isinstance(ns.input_dir, Path)


def test_bool():
    class Options(BaseModel):
        use_alpha: bool = False

    parser = ArgumentParser()
    add_arguments(parser, Options())
    ns = parser.parse_args([])
    assert ns.use_alpha is False
    ns = parser.parse_args(["--use-alpha", "1"])
    assert ns.use_alpha is True


def test_list():
    class Options(BaseModel):
        files: List[str] = []

    parser = ArgumentParser()
    add_arguments(parser, Options())
    ns = parser.parse_args([])
    assert ns.files == []
    ns = parser.parse_args(["--files", "a"])
    assert ns.files == ["a"]
    ns = parser.parse_args(["--files", "b", "c"])
    assert ns.files == ["b", "c"]


def test_child():
    class Options(BaseModel):
        class Train(BaseModel):
            batch_size: int = 100

        train: Train = Train()

    parser = ArgumentParser()
    add_arguments(parser, Options())
    ns = parser.parse_args([])
    assert getattr(ns, "train.batch_size") == 100
    ns = parser.parse_args(["--train-batch-size", "50"])
    assert getattr(ns, "train.batch_size") == 50
