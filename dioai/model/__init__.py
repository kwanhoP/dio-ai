from transformers import PretrainedConfig

from .model import GP2MetaToNoteModel, GPT2BaseModel


class PozalabsModelFactory:
    model_map = {GPT2BaseModel.name: GPT2BaseModel, GP2MetaToNoteModel.name: GP2MetaToNoteModel}

    def create(self, name: str, config: PretrainedConfig):
        model_cls = self.model_map.get(name)
        if model_cls is None:
            raise ValueError(f"`name` should be one of {tuple(self.model_map.keys())}")
        return model_cls(config)


__all__ = ["PozalabsModelFactory"]
