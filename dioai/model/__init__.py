import enum
from pathlib import Path
from typing import Optional, Union

from transformers import PretrainedConfig

from .model import (
    BartDenoisingNoteModel,
    BertForDPR,
    ConditionalRelativeTransformer,
    DPRModel,
    GP2MetaToNoteModel,
    GPT2BaseModel,
    GPT2ChordMetaToNoteModel,
    MusicRagGenerator,
)


class ModelType(enum.Enum):
    HuggingFace = "hf"
    PytorchLightning = "pl"


class PozalabsModelFactory:
    model_map = {
        GPT2BaseModel.name: GPT2BaseModel,
        GP2MetaToNoteModel.name: GP2MetaToNoteModel,
        GPT2ChordMetaToNoteModel.name: GPT2ChordMetaToNoteModel,
        ConditionalRelativeTransformer.name: ConditionalRelativeTransformer,
        BartDenoisingNoteModel.name: BartDenoisingNoteModel,
        BertForDPR.name: BertForDPR,
        DPRModel.name: DPRModel,
        MusicRagGenerator.name: MusicRagGenerator,
    }

    def create(
        self,
        name: str,
        config: Optional[PretrainedConfig] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
    ):
        model_cls = self.model_map.get(name)
        model_type = name[-2:]
        if model_cls is None:
            raise ValueError(f"`name` should be one of {tuple(self.model_map.keys())}")

        if checkpoint_dir is not None:
            if model_type == ModelType.HuggingFace.value:
                return model_cls.from_pretrained(checkpoint_dir)
            elif model_type == ModelType.PytorchLightning.value:
                return model_cls.load_from_checkpoint(checkpoint_dir)
        return model_cls(config)

    def create_rag(
        self,
        name: str,
        config: Optional[PretrainedConfig] = None,
        question_encoder=None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
    ):
        model_cls = self.model_map.get(name)
        model_type = name[-2:]
        if model_cls is None:
            raise ValueError(f"`name` should be one of {tuple(self.model_map.keys())}")

        if checkpoint_dir is not None:
            if model_type == ModelType.HuggingFace.value:
                return model_cls.from_pretrained(checkpoint_dir)
            elif model_type == ModelType.PytorchLightning.value:
                return model_cls.load_from_checkpoint(checkpoint_dir)
        return model_cls(config, question_encoder)


__all__ = ["PozalabsModelFactory"]
