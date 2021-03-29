from .meta import META_PARSERS, BaseMetaParser


class MetaParserFactory:
    registered_meta_parsers = tuple(META_PARSERS.keys())

    def create(self, name: str, *args, **kwargs) -> BaseMetaParser:
        if name not in self.registered_meta_parsers:
            raise ValueError(f"`name` should be one of {self.registered_meta_parsers}")

        return META_PARSERS[name](*args, **kwargs)
