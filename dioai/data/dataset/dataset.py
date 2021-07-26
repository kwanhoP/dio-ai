from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Union

import numpy as np
import torch
from fairseq.data.encoders.utils import get_whole_word_mask
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import Dataset, IterableDataset
from transformers import BertConfig
from transformers.models.dpr.configuration_dpr import DPRConfig

from dioai.config import TransformersConfig
from dioai.data.utils import NoiseArguments, NoiseGenerator, NoteDictionary
from dioai.data.utils.constants import META_OFFSET, NOTE_SEQ_COMPONENTS

from .tf_dataset import (
    BartDenoisingNoteTFDataset,
    BertForDPRTFDataset,
    DPRTFDataset,
    GPT2ChordMetaToNoteTFDataset,
    TransformerDataset,
    gather_files,
)


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
            track_category=self.config.track_category,
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


class RelativeTransformerDataset(IterableDataset):
    def __init__(self, config) -> None:
        self.data_path = config.data_dir
        self.split = config.split
        self.config = config

    def __iter__(self):
        for pth in os.listdir(self.data_path):
            meta, note = self._load_datasets(os.path.join(self.data_path, pth))
            shifted_note = self._shifted_note(note, self.config.sos_token_id)
            dec_input, dec_output, enc_input = self._padding(
                shifted_note,
                note,
                meta,
                self.config.pad_token_id,
                self.config.n_ctx,
                self.config.num_meta,
            )
            data_len = len(dec_input)
            for idx in range(data_len):
                yield (
                    torch.tensor(enc_input[idx]).long(),
                    torch.tensor(dec_input[idx]).long(),
                    torch.tensor(dec_output[idx]).long(),
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

    def _load_datasets(self, data_dir) -> Union[np.ndarray, np.ndarray]:
        """npy 파일에 나눠져있는 meta, note 데이터 load"""

        dataset_paths = list(Path(data_dir).rglob("**/*"))

        input_train_files = gather_files(dataset_paths, prefix=f"input_{self.split}")
        target_train_files = gather_files(dataset_paths, prefix=f"target_{self.split}")

        dataset_files_pair = list(zip(input_train_files, target_train_files))
        random.shuffle(dataset_files_pair)
        for (meta_train_path, note_train_path) in dataset_files_pair:
            meta_features = np.load(meta_train_path, allow_pickle=True)
            note_features = np.load(note_train_path, allow_pickle=True)
            return meta_features, note_features


class BartDenoisingNoteDataset(BaseDataset):
    name = "bart_denoising_note_hf"

    def __init__(
        self, config: TransformersConfig, split: str, training: bool = True, shuffle: bool = False
    ):
        self.config = config
        self.training = training
        self.tf_dataset_build_args = dict(
            batch_size=self.config.batch_size,
            max_length=self.config.model.max_position_embeddings,
            pad_id=self.config.model.pad_token_id,
            training=training,
            shuffle=shuffle,
        )
        self.noise_generator = self._initialize_noise_generator(
            note_dict=self.get_note_dictionary(),
            noise_args=self.config.extra_data_args,
        )
        self.tf_dataset = BartDenoisingNoteTFDataset(
            data_dir=self.config.data_dir,
            split=split,
            noise_generator=self.noise_generator,
        )

    @classmethod
    def get_note_dictionary(cls) -> NoteDictionary:
        """BartDenoisingNoteModel 에서 사용하는 vocabulary"""
        note_dictionary = NoteDictionary()
        note_dictionary.add_note_vocabs(
            note_seq_components=NOTE_SEQ_COMPONENTS,
            use_bos_symbol=True,
            use_mask_symbol=True,
        )
        return note_dictionary

    def _initialize_noise_generator(
        self, note_dict: NoteDictionary, noise_args: NoiseArguments, shuffle: bool = False
    ) -> NoiseGenerator:
        mask_whole_words = (
            get_whole_word_mask(noise_args, note_dict)
            if noise_args.mask_length != "subword"
            else None
        )
        noise_generator = NoiseGenerator(
            vocab=note_dict,
            mask_idx=note_dict.mask_index,
            mask_whole_words=mask_whole_words,
            shuffle=shuffle,
            seed=1203,
            args=noise_args,
        )
        return noise_generator


class BertForDPRDataset(BaseDataset):
    name = "bert_hf"

    def __init__(
        self,
        config: Union[TransformersConfig, BertConfig],
        split: str,
        training: bool = True,
        shuffle: bool = False,
    ):
        super().__init__(config=config, training=training, shuffle=shuffle)
        self.tf_dataset = BertForDPRTFDataset(
            self.config.data_dir,
            split=split,
        )

    def prepare_dataset(self) -> Iterator:
        tf_dataset = self.tf_dataset.build(**self.tf_dataset_build_args)
        for batch in tf_dataset.as_numpy_iterator():
            yield batch


class DPRDataset(BaseDataset):
    name = "dpr_model_hf"

    def __init__(
        self,
        config: Union[TransformersConfig, DPRConfig],
        split: str,
        training: bool = True,
        shuffle: bool = False,
    ):
        self.config = config
        self.training = training
        self.tf_dataset_build_args = dict(
            batch_size=self.config.batch_size,
            max_length=self.config.model.max_position_embeddings,
            pad_id=self.config.model.pad_token_id,
            training=training,
            shuffle=shuffle,
        )
        self.tf_dataset = DPRTFDataset(
            self.config.data_dir,
            split=split,
        )

    def prepare_dataset(self) -> Iterator:
        tf_dataset = self.tf_dataset.build(**self.tf_dataset_build_args)
        for batch in tf_dataset.as_numpy_iterator():
            yield batch


class RagDataset(BaseDataset):
    name = "musicrag_hf"

    def __init__(
        self,
        config,
        split,
        training=True,
        shuffle=False,
    ):
        self.config = config
        self.training = training
        self.tf_dataset_build_args = dict(
            batch_size=self.config.batch_size,
            max_length=self.config.model.question_encoder["max_position_embeddings"],
            pad_id=self.config.model.question_encoder["pad_token_id"],
            training=training,
            shuffle=shuffle,
        )
        self.tf_dataset = DPRTFDataset(
            self.config.data_dir,
            split=split,
            for_rag=True,
        )

    def prepare_dataset(self) -> Iterator:
        tf_dataset = self.tf_dataset.build(**self.tf_dataset_build_args)
        for batch in tf_dataset.as_numpy_iterator():
            yield batch
