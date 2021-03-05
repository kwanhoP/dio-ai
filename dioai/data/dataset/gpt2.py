from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, Generator, Iterator, Union

import tensorflow as tf
import torch
from torch.utils.data import IterableDataset

from .tfrecord import FeatureType, TFRecordDataset


class GPT2Dataset(IterableDataset):
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str,
        min_length: int,
        max_length: int,
        batch_size: int = 2048,
        preprocess: bool = True,
        random_crop_in_train: bool = True,
        shuffle: bool = True,
        bucket_by_sequence: bool = True,
        training: bool = True,
        batch_shuffle_size: int = 512,
        shuffle_buffer_size: int = 10000,
        num_threads: int = cpu_count(),
        pad_id: int = 0,
        **batching_scheme_kwargs,
    ):
        super().__init__()

        self.num_threads = num_threads
        self.pad_id = pad_id
        self.tfrecord_dataset = TFRecordDataset(data_dir, split)
        self.tfrecord_dataset_build_args = dict(
            min_length=min_length,
            max_length=max_length,
            batch_size=batch_size,
            preprocess=preprocess,
            random_crop_in_train=random_crop_in_train,
            shuffle=shuffle,
            bucket_by_sequence=bucket_by_sequence,
            training=training,
            batch_shuffle_size=batch_shuffle_size,
            shuffle_buffer_size=shuffle_buffer_size,
            num_threads=num_threads,
            **batching_scheme_kwargs,
        )

    def __iter__(self) -> Generator[Dict[str, torch.Tensor], None, None]:
        for item in self.prepare_dataset():
            yield item

    def prepare_dataset(self) -> Iterator:
        def _compute_attention_mask(_features: FeatureType) -> FeatureType:
            return compute_attention_mask(_features, pad_id=self.pad_id)

        tf_dataset = self.tfrecord_dataset.build(**self.tfrecord_dataset_build_args)
        tf_dataset = tf_dataset.map(_compute_attention_mask, num_parallel_calls=self.num_threads)
        return tf_dataset.as_numpy_iterator()

    def __getitem__(self, index):
        pass


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
