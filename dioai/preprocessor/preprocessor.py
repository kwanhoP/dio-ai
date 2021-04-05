import abc
import enum
import inspect
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Type, Union

import numpy as np
import parmap

from ..exceptions import UnprocessableMidiError
from . import utils
from .chunk_midi import chunk_midi
from .encoder import BaseMetaEncoder, MidiPerformanceEncoder
from .parser import BaseMetaParser
from .utils import constants
from .utils.container import MidiMeta

MIDI_EXTENSIONS = (".mid", ".MID", ".midi", ".MIDI")


class OutputSubDirName(str, enum.Enum):
    CHUNKED = "chunked"
    PARSED = "parsed"
    TMP = "tmp"
    ENCODE_NPY = "output_npy"
    ENCODE_TMP = "npy_tmp"


@dataclass
class OutputSubDirectory:
    chunked: Union[str, Path]
    parsed: Union[str, Path]
    tmp: Union[str, Path]
    encode_npy: Union[str, Path]
    encode_tmp: Union[str, Path]


def get_output_sub_dir(root_dir: Union[str, Path]) -> OutputSubDirectory:
    result = dict()
    for name, member in OutputSubDirName.__members__.items():
        output_dir = root_dir.joinpath(member.value)
        output_dir.mkdir(exist_ok=True, parents=True)
        result[name.lower()] = output_dir
    return OutputSubDirectory(**result)


@dataclass
class EncodingOutput:
    meta: np.ndarray
    note_sequence: np.ndarray


@dataclass
class ChunkMidiArguments:
    steps_per_sec: int
    longest_allowed_space: int
    minimum_chunk_length: int
    preserve_chord_track: bool = False


@dataclass
class ParseMidiArguments:
    bar_window_size: List[int]
    shift_size: int = 1


class BasePreprocessor(abc.ABC):
    name = ""

    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def preprocess(self, *args, **kwargs):
        raise NotImplementedError


class RedditPreprocessor(BasePreprocessor):
    name = "reddit"

    def __init__(
        self,
        meta_parser: BaseMetaParser,
        meta_encoder: BaseMetaEncoder,
        note_sequence_encoder: MidiPerformanceEncoder,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.meta_parser = meta_parser
        self.meta_encoder = meta_encoder
        self.note_sequence_encoder = note_sequence_encoder

    def preprocess(
        self,
        source_dir: Union[str, Path],
        num_cores: int,
        chunk_midi_arguments: ChunkMidiArguments,
        parse_midi_arguments: ParseMidiArguments,
        val_split_ratio: float = 0.1,
        test_split_ratio: float = 0.1,
        model_name: str = "GPT",
    ) -> None:
        for sub_dir in Path(source_dir).iterdir():
            output_sub_dir = get_output_sub_dir(sub_dir)
            chunk_midi(
                midi_dataset_path=sub_dir,
                chunked_midi_path=output_sub_dir.chunked,
                tmp_midi_dir=output_sub_dir.tmp,
                num_cores=num_cores,
                dataset_name=self.name,
                **asdict(chunk_midi_arguments),
            )
            for window_size in parse_midi_arguments.bar_window_size:
                utils.parse_midi(
                    source_dir=str(output_sub_dir.chunked),
                    num_measures=window_size,
                    shift_size=parse_midi_arguments.shift_size,
                    output_dir=output_sub_dir.parsed,
                    num_cores=num_cores,
                )
            self.export_encoded_midi(
                parsed_midi_dir=output_sub_dir.parsed,
                encode_tmp_dir=output_sub_dir.encode_tmp,
                num_cores=num_cores,
            )
            splits = utils.split_train_val_test(
                *utils.concat_npy(output_sub_dir.encode_tmp, model_name, constants.META_LEN),
                val_ratio=val_split_ratio,
                test_ratio=test_split_ratio,
            )
            for split_name, data_split in splits.items():
                np.save(str(output_sub_dir.encode_npy.joinpath(split_name)), data_split)

    def export_encoded_midi(
        self, parsed_midi_dir: Union[str, Path], encode_tmp_dir: Union[str, Path], num_cores: int
    ) -> None:
        midi_filenames = [
            filename
            for filename in parsed_midi_dir.rglob("**/*")
            if filename.suffix in MIDI_EXTENSIONS
        ]
        _midi_filenames_chunk = np.array_split(np.array(midi_filenames), num_cores)
        midi_filenames_chunk = [arr.tolist() for arr in _midi_filenames_chunk]
        parmap.map(
            self._preprocess_midi_chunk,
            midi_filenames_chunk,
            encode_tmp_dir,
            pm_pbar=True,
            pm_processes=num_cores,
        )

    def _preprocess_midi_chunk(
        self, midi_paths_chunk: Iterable[Union[str, Path]], encode_tmp_dir: Union[str, Path]
    ) -> None:
        encode_tmp_dir = Path(encode_tmp_dir)
        for idx, midi_path in enumerate(midi_paths_chunk):
            try:
                encoding_output = self._preprocess_midi(midi_path)
                np.save(encode_tmp_dir.joinpath(f"input_{idx}"), encoding_output.meta)
                np.save(encode_tmp_dir.joinpath(f"target_{idx}"), encoding_output.note_sequence)
            except UnprocessableMidiError:
                continue

    def _preprocess_midi(self, midi_path: Union[str, Path]):
        encoded_meta = np.array(self._encode_meta(self._parse_meta(midi_path)), dtype=object)
        encoded_note_sequence = np.array(self._encode_note_sequence(midi_path), dtype=object)
        return EncodingOutput(meta=encoded_meta, note_sequence=encoded_note_sequence)

    def _encode_note_sequence(self, midi_path: Union[str, Path]) -> np.ndarray:
        return np.array(self.note_sequence_encoder.encode(midi_path))

    def _encode_meta(self, midi_meta: MidiMeta) -> np.ndarray:
        return np.array(self.meta_encoder.encode(midi_meta))

    def _parse_meta(self, midi_path: Union[str, Path]) -> MidiMeta:
        return self.meta_parser.parse(midi_path)


class Pozalabs2Preprocessor(RedditPreprocessor):
    name = "pozalabs2"


PREPROCESSORS: Dict[str, Type[BasePreprocessor]] = {
    obj.name: obj
    for _, obj in globals().items()
    if inspect.isclass(obj) and issubclass(obj, BasePreprocessor) and not inspect.isabstract(obj)
}


class PreprocessorFactory:
    registered_preprocessors = tuple(PREPROCESSORS.keys())

    def create(self, name: str, *args, **kwargs) -> BasePreprocessor:
        if name not in self.registered_preprocessors:
            raise ValueError(f"`name` should be one of {self.registered_preprocessors}")

        return PREPROCESSORS[name](*args, **kwargs)
