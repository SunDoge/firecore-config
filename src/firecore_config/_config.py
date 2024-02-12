from pydantic import BaseModel
import argparse
from argparse import ArgumentParser
import typing
import logging
from typing import Dict, Any, Type, TypeVar, Optional

logger = logging.getLogger(__name__)


def add_arguments(
    parser: ArgumentParser,
    model: BaseModel,
    name_prefix: str = "-",
    dest_prefix: str = "",
):
    for key, field in model.model_fields.items():
        name = name_prefix + "-" + key.replace("_", "-")
        dest = dest_prefix + "." + key

        logger.debug(f"name={name}, dest={dest}, key={key}, field={field}")

        tp = typing.get_origin(field.annotation)

        logger.debug(f"check type: {tp}")

        if tp is None:  # not typing annotation
            if issubclass(field.annotation, BaseModel):
                add_arguments(
                    parser,
                    field.default,
                    name_prefix=name,
                    dest_prefix=dest,
                )
                continue

            if field.annotation is bool:
                # add --flag and --no-flag
                add_bool_argument(parser, name, dest, field.default)
                continue

            add_typed_argument(parser, name, dest, field.default, field.annotation)
            continue

        if tp is typing.Union:
            print("=" * 100)
            inner = typing.get_args(field.annotation)
            if len(inner) == 2 and type(None) in inner:
                tp2 = get_not_none(inner)
                if tp2 is bool:
                    add_bool_argument(parser, name, dest, field.default)
                else:
                    add_typed_argument(parser, name, dest, field.default, tp2)
                continue

        # import ipdb

        # ipdb.set_trace()


def add_bool_argument(parser: ArgumentParser, name: str, dest: str, default: bool):
    parser.add_argument(
        name,
        action=argparse.BooleanOptionalAction,
        default=default,
        dest=dest,
    )


T = TypeVar("T")


def add_typed_argument(
    parser: ArgumentParser,
    name: str,
    dest: str,
    default: Optional[T],
    type: Type[T],
):
    parser.add_argument(
        name,
        dest=dest,
        default=default,
        type=type,
    )


def get_not_none(xs):
    return [x for x in xs if x is not None][0]


def assign_arguments(options: Dict[str, Any], parsed: Dict[str, Any]):
    for key, value in parsed.items():
        parts = key.split(".")[1:]
        dict_ref = options
        for part in parts[:-1]:
            dict_ref = dict_ref[part]
        dict_ref[parts[-1]] = value


class Options(BaseModel):
    optim: typing.Optional[str] = "sgd"
    bool_value: bool = True

    class Train(BaseModel):
        batch_size: int = 4

    train: Train = Train()

    class Val(BaseModel):
        batch_size: int = 8

    val: Val = Val()


def _main():
    from icecream import ic

    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    add_arguments(parser, Options())
    ns = parser.parse_args()
    ic(ns)
    ic(ns.__dict__)

    options = Options()
    ic(options)
    options_dict = options.model_dump()
    assign_arguments(options_dict, ns.__dict__)
    options2 = Options.model_validate(options_dict)
    ic(options2)


if __name__ == "__main__":
    _main()
