from __future__ import annotations

from multiprocessing import cpu_count
from typing import Dict, Generator, Iterator, List, Optional, Union

import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import Dataset, IterableDataset

from dioai.config import TransformersConfig

from .gpt2_tf import GPT2MetaToNoteTFDataset
from .magenta_tfrecord import BatchingSchemeArgs, FeatureType, MagentaTFRecordDataset


class GPT2BaseDataset(IterableDataset):
    name = "gpt2_base"

    def __init__(
        self,
        config: TransformersConfig,
        split: str,
        preprocess: bool = True,
        random_crop_in_train: bool = True,
        shuffle: bool = True,
        training: bool = True,
        batch_shuffle_size: int = 512,
        shuffle_buffer_size: int = 10000,
        num_threads: int = cpu_count(),
        batching_scheme_args: Optional[BatchingSchemeArgs] = None,
    ):
        super().__init__()

        self.config = config
        self.num_threads = num_threads
        self.training = training

        self.tfrecord_dataset = MagentaTFRecordDataset(
            data_dir=config.data_dir, split=getattr(config, f"{split}_split")
        )
        self.tfrecord_dataset_build_args = dict(
            max_length=self.config.model.n_ctx,
            batch_size=self.config.batch_size,
            pad_id=self.config.model.pad_token_id,
            preprocess=preprocess,
            random_crop_in_train=random_crop_in_train,
            shuffle=shuffle,
            training=training,
            batch_shuffle_size=batch_shuffle_size,
            shuffle_buffer_size=shuffle_buffer_size,
            num_threads=num_threads,
            batching_scheme_args=batching_scheme_args,
        )

    def build(self) -> Union[IterableDataset, Dataset]:
        if self.training:
            return self
        return self.to_dataset()

    def __iter__(self) -> Generator[Dict[str, torch.Tensor], None, None]:
        for item in self.prepare_dataset():
            yield item

    def to_dataset(self) -> Dataset:
        class EvalDataset(Dataset):
            def __init__(self, data: List[Dict[str, torch.Tensor]]):
                super().__init__()
                self.dataset = data

            def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
                return self.dataset[item]

            def __len__(self) -> int:
                return len(self.dataset)

        return EvalDataset(list(self.prepare_dataset()))

    def prepare_dataset(self) -> Iterator:
        def _prepare_inputs(_features: FeatureType) -> FeatureType:
            return prepare_inputs(_features, pad_id=self.config.model.pad_token_id)

        tf_dataset = self.tfrecord_dataset.build(**self.tfrecord_dataset_build_args)
        tf_dataset = tf_dataset.map(_prepare_inputs, num_parallel_calls=self.num_threads)
        return tf_dataset.as_numpy_iterator()

    def __getitem__(self, index):
        pass


def prepare_inputs(features: FeatureType, pad_id: int = 0) -> FeatureType:
    """transformers 라이브러리 학습을 위한 데이터 입력을 준비합니다.
    `attention_mask`와 loss 계산을 위해 필요한, `labels`를 포함한 딕셔너리를 리턴합니다.
    `labels`는 `input_ids`와 같으며, transformers 라이브러리 내부에서 loss 계산 시 shift 됩니다.
    """
    result = compute_attention_mask(features, pad_id)
    # 텐서 복사
    result["labels"] = tf.identity(result["input_ids"])
    return result


def compute_attention_mask(features: FeatureType, pad_id: int = 0) -> FeatureType:
    """tensorflow dataset의 값들을 transformers GPT2 모델에 사용되는 입력값으로 변환합니다.
    현재 리턴값의 키는 `transformers.GPT2Tokenizer.model_inputs_names` (input_ids, attention_mask)입니다.
    """
    targets = features["targets"]
    result = {
        "input_ids": targets,
        "attention_mask": tf.where(
            tf.not_equal(targets, pad_id),
            tf.ones_like(targets, dtype=targets.dtype),
            tf.zeros_like(targets, dtype=targets.dtype),
        ),
    }
    return result


class GPT2MetaToNoteDataset(IterableDataset):
    name = "gpt2_meta_to_note"

    def __init__(
        self, config: TransformersConfig, split: str, training: bool = True, shuffle: bool = False
    ):
        self.config = config
        self.training = training
        self.tf_dataset = GPT2MetaToNoteTFDataset(self.config.data_dir, split=split)
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
        class EvalDataset(Dataset):
            def __init__(self, data: List[Dict[str, torch.Tensor]]):
                super().__init__()
                self.dataset = data

            def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
                return self.dataset[item]

            def __len__(self) -> int:
                return len(self.dataset)

        return EvalDataset(list(self.prepare_dataset()))

    def prepare_dataset(self) -> Iterator:
        tf_dataset = self.tf_dataset.build(**self.tf_dataset_build_args)
        return tf_dataset.as_numpy_iterator()

    def __getitem__(self, item: int):
        ...


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
