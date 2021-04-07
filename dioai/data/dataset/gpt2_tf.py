import copy
from pathlib import Path
from typing import Union

import numpy as np
import tensorflow as tf


class GPT2MetaToNoteTFDataset:
    input_filename = "input"
    target_filename = "target"
    npy_dir_name_prefix = "output_npy"
    output_signature = {
        "input_ids": tf.TensorSpec(shape=(None,), dtype=tf.int64),
        "attention_mask": tf.TensorSpec(shape=(None,), dtype=tf.int64),
        "labels": tf.TensorSpec(shape=(None,), dtype=tf.int64),
    }

    def __init__(self, data_dir: Union[str, Path], split: str):
        self.data_dir = Path(data_dir)
        self.split = split

    def build(
        self,
        batch_size: int,
        max_length: int,
        training: bool = True,
        shuffle: bool = False,
        shuffle_buffer_size: int = 10000,
        pad_id: int = 103,
    ) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_generator(
            self._get_numpy_generator,
            output_signature=self.output_signature,
        )
        dataset = dataset.filter(lambda x: tf.shape(x["input_ids"])[0] <= max_length)
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes={"input_ids": [None], "attention_mask": [None], "labels": [None]},
            padding_values={
                "input_ids": tf.constant(pad_id, dtype=tf.int64),
                "attention_mask": tf.constant(0, dtype=tf.int64),
                "labels": tf.constant(pad_id, dtype=tf.int64),
            },
        )
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer_size)
        if training:
            dataset = dataset.repeat()
            dataset = dataset.shuffle(batch_size)
        else:
            dataset = dataset.shuffle(shuffle_buffer_size, seed=1203)
            dataset = dataset.take(5000)
            dataset = self._get_dataset_with_fixed_batch_size(dataset, batch_size)

        dataset = dataset.prefetch(2)
        return dataset

    def _get_dataset_with_fixed_batch_size(
        self, dataset: tf.data.Dataset, batch_size: int
    ) -> tf.data.Dataset:
        """정확히 `batch_size`인 배치만으로 새로운 `Dataset` 인스턴스를 생성합니다.
        `tf.data.Dataset.padded_batch`가 `batch_size`보다 작은 배치를 리턴할 때가 있는데,
        이 경우 `torch.utils.data.DataLoader`에서 `tf.data.Dataset.unbatch()` 이후
        `batch_size`로 다시 배치를 묶을 때 시퀀스 길이가 다른 배치끼리 배치로 묶여 에러가 발생합니다.
        이 에러를 방지하기 위해 정확히 `batch_size`인 배치만 리턴하도록 합니다.
        """

        def _generator():
            keys = self.output_signature.keys()
            for batch in dataset.as_numpy_iterator():
                # batch: Dict[str, np.ndarray]
                if batch["input_ids"].shape[0] != batch_size:
                    continue

                for i in range(batch_size):
                    yield {key: batch[key][i] for key in keys}

        return tf.data.Dataset.from_generator(_generator, output_signature=self.output_signature)

    def _get_numpy_generator(self):
        for sub_dir in self.data_dir.iterdir():
            if not sub_dir.name.startswith(self.npy_dir_name_prefix):
                continue

            input_features = self.load_npy(sub_dir, is_input=True)
            target_features = self.load_npy(sub_dir, is_input=False)
            for input_feature, target_feature in zip(input_features, target_features):
                input_ids = np.concatenate([input_feature, target_feature])
                attention_mask = compute_attention_mask(input_ids)
                yield {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": copy.deepcopy(input_ids),
                }

    def load_npy(self, source_dir: Union[str, Path], is_input: bool = True) -> np.ndarray:
        return np.load(str(source_dir.joinpath(self._filename(is_input))), allow_pickle=True)

    def _filename(self, is_input: bool = True) -> str:
        return f"{self.input_filename if is_input else self.target_filename}_{self.split}.npy"


def compute_attention_mask(input_features: np.ndarray) -> np.ndarray:
    """attention mask 계산. 입력값은 메타 정보로 패딩이 없습니다."""
    return np.ones_like(input_features)
