from dataclasses import dataclass
from typing import Dict, List, Optional

import fairseq
import numpy as np
import torch
from fairseq.data import DenoisingDataset, data_utils

from dioai.data.utils.constants import NOTE_SEQ_OFFSET


@dataclass
class NoiseArguments:
    mask: float
    poisson_lambda: float
    insert: float
    mask_random: float
    rotate: float
    permute_sentences: float
    replace_length: int
    mask_length: str
    bpe: str


class NoteDictionary(fairseq.data.Dictionary):
    def __init__(
        self,
        pad: str = "<pad>",
        eos: str = "</s>",
        bos: Optional[str] = None,
        unk: Optional[str] = None,
    ):
        self.pad_word, self.eos_word, self.bos_word, self.unk_word = pad, eos, bos, unk
        self.symbols = []
        self.count = []
        self.indices = {}
        self.pad_index = self.add_symbol(pad) if isinstance(pad, str) else pad
        self.eos_index = self.add_symbol(eos) if isinstance(eos, str) else eos
        self.bos_index = self.add_symbol(bos) if isinstance(bos, str) else bos
        self.unk_index = self.add_symbol(unk) if isinstance(unk, str) else unk
        self.mask_index = None
        self.nspecial = len(self.symbols)

    def add_note_vocabs(
        self,
        note_seq_components: Dict[str, int],
        use_bos_symbol: bool = True,
        use_mask_symbol: bool = True,
    ):
        assert len(self.symbols) == NOTE_SEQ_OFFSET
        for note_seq_component, size in note_seq_components.items():
            for idx in range(size):
                self.add_symbol(f"{note_seq_component}_{idx}")
        if use_bos_symbol:
            self.bos_index = self.add_symbol("<s>")
        if use_mask_symbol:
            self.mask_index = self.add_symbol("<mask>")

        self.nspecial = len(self.symbols) - sum(note_seq_components.values())


class NoiseGenerator(DenoisingDataset):
    def __init__(self, dataset: torch.Tensor = None, sizes: List[int] = None, **kwargs):
        super().__init__(dataset, sizes, **kwargs)

    def on_dataset(self, dataset: List[int]):
        self.dataset = torch.tensor([dataset])
        return self

    def __getitem__(self, index):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            tokens = self.dataset[index]
            assert tokens[-1] == self.eos
            source, target = tokens, tokens.clone()

            if self.permute_sentence_ratio > 0.0:
                source = self.permute_sentences(source, self.permute_sentence_ratio)

            if self.insert_ratio > 0:
                source = self.add_insertion_noise(source, self.insert_ratio)

            if self.mask_ratio > 0:
                source = self.add_whole_word_mask(source, self.mask_ratio)

            if self.rotate_ratio > 0.0 and np.random.random() < self.rotate_ratio:
                source = self.add_rolling_noise(source)
        # there can additional changes to make:
        if self.item_transform_func is not None:
            source, target = self.item_transform_func(source, target)

        assert (source >= 0).all()
        assert (source[1:-1] >= 1).all()
        assert (source <= len(self.vocab)).all()
        assert source[0] == self.vocab.bos()
        assert source[-1] == self.eos
        return source
