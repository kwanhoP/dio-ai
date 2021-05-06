from __future__ import annotations

from typing import Dict, Generator, Iterator, List, Union

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from dioai.config import TransformersConfig

from .tf_dataset import GPT2ChordMetaToNoteTFDataset, TransformerDataset


class EvalDataset(Dataset):
    def __init__(self, data: List[Dict[str, torch.Tensor]]):
        super().__init__()
        self.dataset = data

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        return self.dataset[item]

    def __len__(self) -> int:
        return len(self.dataset)


class BaseDataset(IterableDataset):
    name = "base"

    def __init__(self, config: TransformersConfig, training: bool = True, shuffle: bool = False):
        self.config = config
        self.training = training
        self.tf_dataset_build_args = dict(
            batch_size=self.config.batch_size,
            max_length=self.config.model.n_ctx,
            pad_id=self.config.model.pad_token_id,
            training=training,
            shuffle=shuffle,
        )

    def build(self) -> Union[IterableDataset, Dataset]:
        if self.training:
            return self
        return self.to_dataset()

    def __iter__(self) -> Generator[Dict[str, torch.Tensor], None, None]:
        for item in self.prepare_dataset():
            yield item

    def to_dataset(self) -> Dataset:
        return EvalDataset(list(self.prepare_dataset()))

    def prepare_dataset(self) -> Iterator:
        tf_dataset = self.tf_dataset.build(**self.tf_dataset_build_args)
        return tf_dataset.as_numpy_iterator()


class GPT2ChordMetaToNoteDataset(BaseDataset):
    name = "gpt2_chord_meta_to_note"

    def __init__(
        self, config: TransformersConfig, split: str, training: bool = True, shuffle: bool = False
    ):
        super().__init__(config=config, training=training, shuffle=shuffle)
        self.tf_dataset = GPT2ChordMetaToNoteTFDataset(
            self.config.data_dir,
            split=split,
            chord_embedding_path=self.config.chord_embedding_path,
            num_meta=self.config.num_meta,
        )

    def prepare_dataset(self) -> Iterator:
        tf_dataset = self.tf_dataset.build(**self.tf_dataset_build_args)
        for batch in tf_dataset.as_numpy_iterator():
            # 반드시 1차원 이상이어야 함
            batch["num_meta"] = np.array([self.config.num_meta], dtype=np.int32)
            yield batch


class Seq2SeqDataset(BaseDataset):
    name = "seq2seq"

    def __init__(
        self, config: TransformersConfig, split: str, training: bool = True, shuffle: bool = False
    ):
        super().__init__(config=config, training=training, shuffle=shuffle)
        self.tf_dataset = TransformerDataset(
            self.config.data_dir,
            split=split,
            chord_embedding_path=self.config.chord_embedding_path,
            num_meta=self.config.num_meta,
        )

    def prepare_dataset(self) -> Iterator:
        tf_dataset = self.tf_dataset.build(**self.tf_dataset_build_args)
        for batch in tf_dataset.as_numpy_iterator():
            yield batch


def meta_to_note_collate_fn(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    def _pad_label(_tensor, _max_length):
        return torch.cat([_tensor, _tensor.new_zeros(_max_length - _tensor.size(0))])

    label_list = [torch.LongTensor(b["labels"]) for b in batch]
    max_length = max(len(label) for label in label_list)
    labels = torch.stack([_pad_label(label, max_length) for label in label_list])

    return {
        "input_ids": torch.stack([torch.LongTensor(b["input_ids"]) for b in batch]),
        "attention_mask": torch.stack([torch.LongTensor(b["attention_mask"]) for b in batch]),
        "labels": labels,
    }
