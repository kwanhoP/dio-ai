from pathlib import Path
from typing import Union

import numpy as np
import tensorflow as tf


class GPT2MetaToNoteTFDataset:
    input_filename = "input"
    target_filename = "target"
    npy_dir_name = "output_npy"

    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)

    def build(
        self,
        batch_size: int,
        training: bool = True,
        shuffle: bool = False,
        shuffle_buffer_size: int = 10000,
        pad_id: int = 0,
    ) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_generator(
            self._get_numpy_generator,
            output_signature={
                "input_ids": tf.TensorSpec(shape=(None,), dtype=tf.int64),
                "attention_mask": tf.TensorSpec(shape=(None,), dtype=tf.int64),
                "labels": tf.TensorSpec(shape=(None,), dtype=tf.int64),
            },
        )
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes={"input_ids": [None], "attention_mask": [None], "labels": [None]},
            padding_values=tf.constant(pad_id, dtype=tf.int64),
        )
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer_size)
        if training:
            dataset = dataset.repeat()
            dataset = dataset.shuffle(batch_size)
        else:
            dataset = dataset.unbatch()
        dataset = dataset.prefetch(2)
        return dataset

    def _get_numpy_generator(self):
        for sub_dir in self.data_dir.glob("**"):
            if sub_dir.name != self.npy_dir_name:
                continue

            def _get_filename(_is_input):
                return str(sub_dir.joinpath(self._filename(_is_input)))

            input_features = np.load(_get_filename(True), allow_pickle=True)
            target_features = np.load(_get_filename(False), allow_pickle=True)
            for input_feature, target_feature in zip(input_features, target_features):
                attention_mask = compute_attention_mask(input_feature)
                yield {
                    "input_ids": input_feature,
                    "attention_mask": attention_mask,
                    "labels": target_feature,
                }

    def _filename(self, is_input: bool = True) -> str:
        return f"{self.input_filename if is_input else self.target_filename}.npy"


def compute_attention_mask(input_features: np.ndarray) -> np.ndarray:
    """attention mask 계산. 입력값은 메타 정보로 패딩이 없습니다."""
    return np.ones_like(input_features)
