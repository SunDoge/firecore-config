from pydantic import BaseModel
import argparse
from argparse import ArgumentParser
import typing
import logging
from typing import Dict, Any, Type, TypeVar, Optional, Union

logger = logging.getLogger(__name__)


_NoneType = type(None)


def str_to_bool(x: Union[str, bool]) -> bool:
    if isinstance(x, bool):
        return x
    x_lower = x.lower()
    if x_lower in ["1", "yes", "true", "t", "y"]:
        return True
    elif x_lower in ["0", "no", "false", "f", "n"]:
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def add_arguments(
    parser: ArgumentParser,
    model: BaseModel,
    name_prefix: str = "-",
    dest_prefix: str = "",
):
    group = parser.add_argument_group(title=f"{model.__class__.__qualname__}")
    for key, field in model.model_fields.items():
        name = name_prefix + "-" + key.replace("_", "-")
        dest = (dest_prefix + "." + key) if dest_prefix else key

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

            # if field.annotation is bool:
            #     # add --flag and --no-flag
            #     # add_bool_argument(group, name, dest, field.default)
            #     continue

            add_typed_argument(group, name, dest, field.default, field.annotation)
            continue

        if tp is typing.Union:
            inner = typing.get_args(field.annotation)
            if len(inner) == 2 and _NoneType in inner:  # typing.Optional
                tp2 = get_not_none(inner)
                if tp2 is bool:
                    add_bool_argument(group, name, dest, field.default)
                else:
                    add_typed_argument(group, name, dest, field.default, tp2)
                continue

        if tp is list:
            inner = typing.get_args(field.annotation)
            assert len(inner) == 1
            tp2 = inner[0]
            add_typed_list_argument(group, name, dest, field.default, tp2)
            continue


def add_bool_argument(parser: ArgumentParser, name: str, dest: str, default: bool):
    parser.add_argument(
        name,
        action=argparse.BooleanOptionalAction,
        default=default,
        dest=dest,
        help=f"(default: bool = {default})",
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
        type=str_to_bool if type is bool else type,
        help=f"(default: {type.__name__} = {default})",
    )


def add_typed_list_argument(
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
        type=str_to_bool if type is bool else type,
        help=f"(default: List[{type.__name__}] = {default})",
        nargs=argparse.ZERO_OR_MORE,
    )


def get_not_none(xs):
    rest = [x for x in xs if x is not _NoneType]
    logger.debug(f"rest: {rest}")
    return rest[0]


def assign_arguments(options: Dict[str, Any], parsed: Dict[str, Any]):
    for key, value in parsed.items():
        parts = key.split(".")
        dict_ref = options
        for part in parts[:-1]:
            dict_ref = dict_ref[part]
        dict_ref[parts[-1]] = value


class Hparams(BaseModel):
    optim: typing.Optional[str] = "sgd"
    bool_value: bool = True

    class Train(BaseModel):
        batch_size: int = 4
        file_list: typing.List[bool] = []

    train: Train = Train()

    class Val(BaseModel):
        batch_size: int = 8

    val: Val = Val()


def _main():
    from icecream import ic

    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dv", default=1)
    add_arguments(parser, Hparams())
    ns = parser.parse_args()
    ic(ns)
    ic(ns.__dict__)

    options = Hparams()
    ic(options)
    options_dict = options.model_dump()
    assign_arguments(options_dict, ns.__dict__)
    options2 = Hparams.model_validate(options_dict)
    ic(options2)


if __name__ == "__main__":
    _main()
