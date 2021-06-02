import inspect
import pprint

from dioai.config import TransformersConfig

from .dataset import (
    BartDenoisingNoteDataset,
    BertForDPRDataset,
    GPT2ChordMetaToNoteDataset,
    Seq2SeqDataset,
    meta_to_note_collate_fn,
)


class PozalabsDatasetFactory:
    dataset_map = {
        GPT2ChordMetaToNoteDataset.name: GPT2ChordMetaToNoteDataset,
        Seq2SeqDataset.name: Seq2SeqDataset,
        BartDenoisingNoteDataset.name: BartDenoisingNoteDataset,
        BertForDPRDataset.name: BertForDPRDataset,
    }

    def create(self, config: TransformersConfig, *args, **kwargs):
        dataset_cls = self.dataset_map.get(config.model_name)
        if dataset_cls is None:
            raise ValueError(f"`name` should be one of {tuple(self.dataset_map.keys())}")
        try:
            instance = dataset_cls(config=config, *args, **kwargs)
        except TypeError as exc:
            full_arg_spec = inspect.getfullargspec(dataset_cls.__init__)
            raise TypeError(
                f"Failed to initialize {dataset_cls.__class__.__name__} due to invalid arguments. "
                f"Check all required arguments are passed. Required args:\n"
                f"{pprint.pformat(full_arg_spec.annotations, indent=2)}"
            ) from exc
        return instance.build()


__all__ = ["GPT2BaseDataset", "PozalabsDatasetFactory", "meta_to_note_collate_fn"]
