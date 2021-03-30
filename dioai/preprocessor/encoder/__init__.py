from .encoder import *  # noqa: F403, F401
from .meta import META_ENCODERS, BaseMetaEncoder


class MetaEncoderFactory:
    registered_meta_encoders = tuple(META_ENCODERS.keys())

    def create(self, name: str, *args, **kwargs) -> BaseMetaEncoder:
        if name not in self.registered_meta_encoders:
            raise ValueError(f"`name` should be one of {self.registered_meta_encoders}")

        return META_ENCODERS[name](*args, **kwargs)
