from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Union

import numpy as np
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import Dataset, IterableDataset

from dioai.config import TransformersConfig

from .tf_dataset import GPT2ChordMetaToNoteTFDataset, TransformerDataset, gather_files

META_OFFSET = 421


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
    name = "gpt2_chord_meta_to_note_hf"

    def __init__(
        self, config: TransformersConfig, split: str, training: bool = True, shuffle: bool = False
    ):
        super().__init__(config=config, training=training, shuffle=shuffle)
        self.tf_dataset = GPT2ChordMetaToNoteTFDataset(
            self.config.data_dir,
            split=split,
            chord_embedding_path=self.config.chord_embedding_path,
            num_meta=self.config.num_meta,
            n_embed=self.config.model.n_embd,
        )

    def prepare_dataset(self) -> Iterator:
        tf_dataset = self.tf_dataset.build(**self.tf_dataset_build_args)
        for batch in tf_dataset.as_numpy_iterator():
            # 반드시 1차원 이상이어야 함
            batch["num_meta"] = np.array([self.config.num_meta], dtype=np.int64)
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


class RelativeTransformerDataset(torch.utils.data.Dataset):
    """relative self attention을 사용하는 transformer dataset
    music transformer와 다르게 인코더, 디코더 인풋이 다름(meta, note)

    Args:
        config: config.PytorchlightConfig
    """

    def __init__(self, config):
        self.split = config.split
        self.data_dir = config.data_dir
        self.pad_token = config.pad_token_id
        self.sos_token = config.sos_token_id
        self.maxlen = config.n_ctx
        self.num_meta = config.num_meta
        self.data_generator = self.dataset_generator(self.data_dir)
        self.meta, self.note, self.data_len = self._load_dataset(self.data_generator)
        self.shifted_note = self._shifted_note(self.note, self.sos_token)
        self.dec_input, self.dec_output, self.enc_input = self._padding(
            self.shifted_note, self.note, self.meta, self.pad_token, self.maxlen, self.num_meta
        )
        self.total_data_len = config.max_steps * config.batch_size * config.n_gpu

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        if idx >= self.data_len:
            pre_data_len = self.data_len
            self.meta, self.note, self.data_len = self._load_dataset(self.data_generator)
            self.data_len = self.data_len + pre_data_len
            self.shifted_note = self._shifted_note(self.note, self.sos_token)
            self.dec_input, self.dec_output, self.enc_input = self._padding(
                self.shifted_note, self.note, self.meta, self.pad_token, self.maxlen, self.num_meta
            )

            return (
                torch.tensor(self.enc_input[max(0, idx - self.data_len)]).long(),
                torch.tensor(self.dec_input[max(0, idx - self.data_len)]).long(),
                torch.tensor(self.dec_output[max(0, idx - self.data_len)]).long(),
            )
        else:
            return (
                torch.tensor(self.enc_input[idx]).long(),
                torch.tensor(self.dec_input[idx]).long(),
                torch.tensor(self.dec_output[idx]).long(),
            )

    def _padding(
        self,
        shifted_note: np.ndarray,
        note: np.ndarray,
        meta: np.ndarray,
        pad_token: int,
        maxlen: int,
        num_meta: int,
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        """meta, note 각각 길이 동일하게 padding

        Args:
            shifted_note: _shifted_note 함수 return 값
            pad_token: 0(default)
            maxlen: note_seqeunce 최대 길이(default: 512)
            num_meta: meta 정보 길이

        Return:
            패딩 된 shifted_note, note, meta

        To do: 코드 진행 임배딩 반영
        """
        note_pad = pad_sequences(note, maxlen=maxlen, padding="post", value=pad_token)
        shifted_note_pad = pad_sequences(
            shifted_note, maxlen=maxlen, padding="post", value=pad_token
        )
        meta_new = []
        for i in meta:
            meta_new.append((i[:-1].astype(np.int64)) - META_OFFSET)
        meta_pad = pad_sequences(meta_new, maxlen=num_meta, padding="post", value=pad_token)
        return shifted_note_pad, note_pad, meta_pad

    def _shifted_note(self, data: np.ndarray, sos_token: int) -> np.ndarray:
        """auto regressive한 모델을 위해 dec_input shift & sos 토큰 추가

        Args:
            data: note_sequence
            sos_token: start token(default: 422)

        Returns:
            shifted_note_sequence
        """
        shifted = []
        for sample in data:
            shifted.append(np.insert(sample, 0, sos_token))
        return np.array(shifted)

    def dataset_generator(self, data_dir) -> Union[np.ndarray, np.ndarray]:
        """여러 npy 파일에 나눠져있는 데이터를 하나씩 불러와 사용하기 위한 generator"""

        dataset_paths = list(Path(data_dir).rglob("**/*"))

        input_train_files = gather_files(dataset_paths, prefix=f"input_{self.split}")
        target_train_files = gather_files(dataset_paths, prefix=f"target_{self.split}")

        dataset_files_pair = list(zip(input_train_files, target_train_files))
        random.shuffle(dataset_files_pair)
        for (meta_train_path, note_train_path) in dataset_files_pair:
            meta_features = np.load(meta_train_path, allow_pickle=True)
            note_features = np.load(note_train_path, allow_pickle=True)
            yield meta_features, note_features

    def _load_dataset(self, data_generator):
        meta, note = next(data_generator)

        return meta, note, len(meta)
