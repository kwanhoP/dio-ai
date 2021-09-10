import enum
from pathlib import Path
from typing import Optional, Union

from transformers import PretrainedConfig

from .model import (
    BartDenoisingNoteModel,
    BertForDPR,
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
    BartPretrainedRag = "bart_pretrained_hf"
    model_map = {
        GPT2BaseModel.name: GPT2BaseModel,
        GP2MetaToNoteModel.name: GP2MetaToNoteModel,
        GPT2ChordMetaToNoteModel.name: GPT2ChordMetaToNoteModel,
        BartDenoisingNoteModel.name: BartDenoisingNoteModel,
        BartPretrainedRag: BartDenoisingNoteModel,
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
        generator=None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
    ):
        model_cls = self.model_map.get(name)
        model_type = name[-2:]
        if model_cls is None:
            raise ValueError(f"`name` should be one of {tuple(self.model_map.keys())}")

        if checkpoint_dir is not None:
            if model_type == ModelType.HuggingFace.value:
                return model_cls.from_pretrained(
                    checkpoint_dir, question_encoder=question_encoder, generator=generator
                )
            elif model_type == ModelType.PytorchLightning.value:
                return model_cls.load_from_checkpoint(checkpoint_dir)
        return model_cls(config, question_encoder, generator)

    def create_dpr(
        self,
        name: str,
        config: Optional[PretrainedConfig] = None,
        pretrained_bert=None,
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
        return model_cls(config, pretrained_bert)


__all__ = ["PozalabsModelFactory"]
