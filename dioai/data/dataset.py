import random
from dataclasses import asdict, dataclass, field
from multiprocessing import cpu_count
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Union

import tensorflow as tf
from tensorflow.python.ops import parsing_ops

FeatureType = Dict[str, tf.Tensor]


@dataclass
class BatchingScheme:
    boundaries: List[int]
    batch_sizes: List[int]


@dataclass
class BatchingSchemeArgs:
    max_length: int
    num_tokens_batch: int = field(default=2048)
    min_length_bucket: int = field(default=8)
    length_bucket_step: float = field(default=1.5)
    shard_multiplier: int = field(default=1)
    length_multiplier: int = field(default=1)
    min_length: int = field(default=0)


class TFRecordDataset:
    """Magenta (Tensor2Tensor)를 사용해 제작된 tfrecord 파일을 읽기 위한 객체.
    `tf.train.Example` 형식으로 저장된 tfrecord 파일을 읽습니다.
    """

    allowed_splits = ("train", "dev", "test")

    def __init__(self, data_dir: Union[Path, str], split: str):
        if split not in self.allowed_splits:
            raise ValueError(f"`split` should be one of {self.allowed_splits}")

        self.data_dir = Path(data_dir)
        self.split = split

    def build(
        self,
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
        **batching_scheme_kwargs,
    ) -> tf.data.Dataset:
        files = [
            str(filename) for filename in self.data_dir.rglob("**/*") if self.split in filename.name
        ]
        if not files:
            raise ValueError(f"Could not find any files for split: {self.split}")

        if shuffle:
            random.shuffle(files)

        dataset = tf.data.TFRecordDataset(files)
        dataset = dataset.map(self.decode_example, num_parallel_calls=num_threads)
        if preprocess:
            dataset = self.preprocess(
                dataset,
                max_sequence_length=max_length,
                random_crop_in_train=random_crop_in_train,
                interleave=shuffle,
            )
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer_size)
        if training:
            dataset = dataset.repeat()
        dataset = dataset.map(int_to_int32, num_parallel_calls=num_threads)
        dataset = dataset.map(to_dense, num_parallel_calls=num_threads)
        dataset = dataset.filter(valid_size(min_length, max_length))

        if bucket_by_sequence:
            batching_scheme_args = BatchingSchemeArgs(
                **{**batching_scheme_kwargs, "min_length": min_length, "max_length": max_length}
            )
            scheme = batching_scheme(**asdict(batching_scheme_args))
            dataset = dataset.apply(
                tf.data.experimental.bucket_by_sequence_length(
                    example_length,
                    bucket_boundaries=scheme.boundaries,
                    bucket_batch_sizes=scheme.batch_sizes,
                )
            )
        else:
            dataset = dataset.batch(batch_size)

        if training and batch_shuffle_size:
            dataset = dataset.shuffle(batch_shuffle_size)

        dataset = dataset.prefetch(2)
        return dataset

    def preprocess(
        self,
        dataset: tf.data.Dataset,
        max_sequence_length: int,
        random_crop_in_train: bool = True,
        interleave: bool = True,
    ) -> tf.data.Dataset:
        def _preprocess_example(example):
            return self.preprocess_example(
                example,
                max_sequence_length=max_sequence_length,
                random_crop_in_train=random_crop_in_train,
            )

        if interleave:
            return dataset.interleave(_preprocess_example, cycle_length=8)
        return dataset.flat_map(_preprocess_example)

    def preprocess_example(
        self, example: FeatureType, max_sequence_length: int, random_crop_in_train: bool = True,
    ) -> tf.data.Dataset:
        if self.split == "train" and random_crop_in_train:
            max_offset = tf.maximum(tf.shape(example["targets"])[0] - max_sequence_length, 0)
            offset = tf.cond(
                max_offset > 0,
                true_fn=lambda: tf.random.uniform([], maxval=max_offset, dtype=tf.int32),
                false_fn=lambda: 0,
            )
            example["targets"] = example["targets"][offset : offset + max_sequence_length]
        return tf.data.Dataset.from_tensors(example)

    def decode_example(self, serialized_example: bytes) -> FeatureType:
        return parsing_ops.parse_single_example(
            serialized=serialized_example, features=self.example_reading_spec()
        )

    @staticmethod
    def example_reading_spec() -> Dict[str, NamedTuple]:
        """`tf.train.Example`을 읽기 위한 스키마 정의.
        `tf.io.*Feature`는 `collections.namedtuple`를 상속합니다.
        """
        return {"targets": tf.io.VarLenFeature(tf.int64)}


def preprocess_example_common(
    example: FeatureType, max_input_sequence_length: int, max_target_sequence_length: int
) -> Dict[str, tf.Tensor]:
    if "inputs" in example and max_input_sequence_length > 0:
        example["inputs"] = example["inputs"][:max_input_sequence_length]

    if "targets" in example and max_target_sequence_length > 0:
        example["targets"] = example["targets"][:max_target_sequence_length]

    return example


def valid_size(min_length: int, max_length: int) -> Callable[[FeatureType], bool]:
    def get_length(_example):
        _length = 0
        for _, value in _example.items():
            feature_length = tf.shape(value)[0]
            _length = tf.maximum(_length, feature_length)
        return _length

    def _valid_size(_example):
        length = get_length(_example)
        return tf.logical_and(length >= min_length, length <= max_length)

    return _valid_size


def int_to_int32(features: FeatureType) -> FeatureType:
    result = dict()
    for key, value in features.items():
        if value.dtype in (tf.int64, tf.uint8):
            value = tf.cast(value, tf.int32)
        result[key] = value
    return result


def to_dense(features: FeatureType) -> FeatureType:
    result = dict()
    for key, value in features.items():
        if isinstance(value, tf.SparseTensor):
            value = tf.sparse.to_dense(value)
        result[key] = value
    return result


def example_length(example: FeatureType) -> int:
    length = 0
    # Length of the example is the maximum length of the feature lengths
    for _, v in example.items():
        # For images the sequence length is the size of the spatial dimensions.
        feature_length = tf.shape(v)[0]
        if len(v.get_shape()) > 2:
            feature_length = tf.shape(v)[0] * tf.shape(v)[1]
        length = tf.maximum(length, feature_length)
    return length


def batching_scheme(
    num_tokens_batch: int,
    min_length_bucket: int,
    length_bucket_step: float,
    shard_multiplier: int = 1,
    length_multiplier: int = 1,
    min_length: int = 0,
    max_length: Optional[int] = None,
) -> BatchingScheme:
    """`tf.data.experimental.bucket_by_sequence_length` 함수에 사용할 인자들을 리턴합니다.
    아래 코드를 수정해서 사용합니다.
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/data_reader.py#L81

    Args:
        num_tokens_batch: `int`. 배치 내의 총 토큰 개수
        min_length_bucket: `int`. 버켓 바운더리 최솟값
        length_bucket_step: `float`. 버켓 바운더리를 계산에 사용되는 배수
        shard_multiplier: `int`. 버켓별로 배치를 나눌 때 사용되는 배수
        length_multiplier: `int`. 배치 크기와 시퀀스 길이를 늘리는데 사용되는 배수
        min_length: `int` 배치에 포함할 시퀀스의 최소 길이 (default: 0)
        max_length: `Optional[int]`. 배치에 포함할 시퀀스의 최대 길이 (default: `num_tokens_batch`)
    """
    if length_bucket_step < 1.0:
        raise ValueError("`length_bucket_step` must be greater than 1.0")

    max_length = max_length or num_tokens_batch
    if max_length < min_length:
        raise ValueError("max_length must be greater or equal to min_length")

    boundaries = _bucket_boundaries(max_length, min_length_bucket, length_bucket_step)
    boundaries = [boundary * length_multiplier for boundary in boundaries]
    max_length *= length_multiplier

    batch_sizes = [max(1, num_tokens_batch // length) for length in boundaries + [max_length]]
    max_batch_size = max(batch_sizes)
    # Since the Datasets API only allows a single constant for window_size,
    # and it needs divide all bucket_batch_sizes, we pick a highly-composite
    # window size and then round down all batch sizes to divisors of that window
    # size, so that a window can always be divided evenly into batches.
    # TODO(noam): remove this when Dataset API improves.
    # fmt: off
    highly_composite_numbers = [
        1, 2, 4, 6, 12, 24, 36, 48, 60, 120, 180, 240, 360, 720, 840, 1260, 1680,
        2520, 5040, 7560, 10080, 15120, 20160, 25200, 27720, 45360, 50400, 55440,
        83160, 110880, 166320, 221760, 277200, 332640, 498960, 554400, 665280,
        720720, 1081080, 1441440, 2162160, 2882880, 3603600, 4324320, 6486480,
        7207200, 8648640, 10810800, 14414400, 17297280, 21621600, 32432400,
        36756720, 43243200, 61261200, 73513440, 110270160
    ]
    # fmt: on
    window_size = max([i for i in highly_composite_numbers if i <= 3 * max_batch_size])
    divisors = [i for i in range(1, window_size + 1) if window_size % i == 0]
    batch_sizes = [max([d for d in divisors if d <= bs]) for bs in batch_sizes]
    window_size *= shard_multiplier
    batch_sizes = [bs * shard_multiplier for bs in batch_sizes]
    return BatchingScheme(boundaries=boundaries, batch_sizes=batch_sizes)


def _bucket_boundaries(
    max_length: int, min_length: int = 8, length_bucket_step: float = 1.1
) -> List[int]:
    """A default set of length-bucket boundaries."""
    assert length_bucket_step > 1.0
    x = min_length
    boundaries = []
    while x < max_length:
        boundaries.append(x)
        x = max(x + 1, int(x * length_bucket_step))
    return boundaries
