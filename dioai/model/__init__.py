from pathlib import Path
from typing import Optional, Union

from transformers import PretrainedConfig

from .model import GP2MetaToNoteModel, GPT2BaseModel, GPT2ChordMetaToNoteModel


class PozalabsModelFactory:
    model_map = {
        GPT2BaseModel.name: GPT2BaseModel,
        GP2MetaToNoteModel.name: GP2MetaToNoteModel,
        GPT2ChordMetaToNoteModel.name: GPT2ChordMetaToNoteModel,
    }

    def create(
        self,
        name: str,
        config: Optional[PretrainedConfig] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
    ):
        model_cls = self.model_map.get(name)
        if model_cls is None:
            raise ValueError(f"`name` should be one of {tuple(self.model_map.keys())}")

        if checkpoint_dir is not None:
            return model_cls.from_pretrained(checkpoint_dir)
        return model_cls(config)


__all__ = ["PozalabsModelFactory"]
