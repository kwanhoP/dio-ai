from typing import Any, Dict

from .meta import META_PARSERS, BaseMetaParser


class MetaParserFactory:
    registered_meta_parsers = tuple(META_PARSERS.keys())
    meta_parser_args = {
        "reddit": ("default_to_unknown",),
        "pozalabs": None,
        "pozalabs2": ("meta_csv_path",),
    }

    def create(self, name: str, **kwargs) -> BaseMetaParser:
        if name not in self.registered_meta_parsers:
            raise ValueError(f"`name` should be one of {self.registered_meta_parsers}")

        cls_args = self.meta_parser_args[name]
        parser_kwargs = get_kwargs(cls_args, **kwargs) if cls_args is not None else dict()
        return META_PARSERS[name](**parser_kwargs)


def get_kwargs(args, **kwargs) -> Dict[str, Any]:
    return {key: value for key, value in kwargs.items() if key in args}
