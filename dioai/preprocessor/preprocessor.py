import abc
import copy
import enum
import gc
import inspect
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

import mido
import numpy as np
import pandas as pd
import parmap

from ..exceptions import UnprocessableMidiError
from ..utils import dependency
from . import utils
from .chunk_midi import chunk_midi
from .encoder import BaseMetaEncoder, MidiPerformanceEncoder
from .parser import BaseMetaParser
from .utils import constants
from .utils.container import MidiMeta

MIDI_EXTENSIONS = (".mid", ".MID", ".midi", ".MIDI")


class OutputSubDirName(str, enum.Enum):
    RAW = "raw"
    CHUNKED = "chunked"
    PARSED = "parsed"
    TMP = "tmp"
    ENCODE_NPY = "output_npy"
    ENCODE_TMP = "npy_tmp"


class SubDirName(str, enum.Enum):
    RAW = "raw"
    CHUNKED = "chunked"
    PARSED = "parsed"
    TMP = "tmp"
    ENCODE_NPY = "output_npy"
    ENCODE_TMP = "npy_tmp"
    AUGMENTED_TMP = "augmented_tmp"
    AUGMENTED = "augmented"


@dataclass
class OutputSubDirectory:
    chunked: Union[str, Path]
    parsed: Union[str, Path]
    tmp: Union[str, Path]
    encode_npy: Union[str, Path]
    encode_tmp: Union[str, Path]


@dataclass
class SubDirectory:
    raw: Union[str, Path]
    encode_npy: Union[str, Path]
    encode_tmp: Union[str, Path]
    chunked: Optional[Union[str, Path]] = field(default=None)
    parsed: Optional[Union[str, Path]] = field(default=None)
    tmp: Optional[Union[str, Path]] = field(default=None)
    augmented_tmp: Optional[Union[str, Path]] = field(default=None)
    augmented: Optional[Union[str, Path]] = field(default=None)


def get_output_sub_dir(root_dir: Union[str, Path]) -> OutputSubDirectory:
    result = dict()
    for name, member in OutputSubDirName.__members__.items():
        output_dir = root_dir.joinpath(member.value)
        output_dir.mkdir(exist_ok=True, parents=True)
        result[name.lower()] = output_dir
    return OutputSubDirectory(**result)


def get_sub_dir(
    root_dir: Union[str, Path], exclude: Optional[Iterable[str]] = None
) -> SubDirectory:
    result = dict()
    for name, member in SubDirName.__members__.items():
        sub_dir = root_dir.joinpath(member.value)
        if exclude is not None:
            if member.value in exclude:
                continue
            sub_dir.mkdir(exist_ok=True, parents=True)
        else:
            sub_dir.mkdir(exist_ok=True, parents=True)
        result[name.lower()] = sub_dir
    return SubDirectory(**result)


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
    preserve_channel: bool = False


@dataclass
class ParseMidiArguments:
    bar_window_size: List[int]
    shift_size: int = 1
    preserve_channel: bool = False


class BasePreprocessor(abc.ABC):
    name = ""

    def __init__(
        self,
        meta_parser: BaseMetaParser,
        meta_encoder: BaseMetaEncoder,
        note_sequence_encoder: MidiPerformanceEncoder,
        *args,
        **kwargs,
    ):
        self.meta_parser = meta_parser
        self.meta_encoder = meta_encoder
        self.note_sequence_encoder = note_sequence_encoder

    @abc.abstractmethod
    def preprocess(self, *args, **kwargs):
        raise NotImplementedError

    def augment_data(
        self,
        source_dir: Union[str, Path],
        augmented_dir: Union[str, Path],
        augmented_tmp_dir: Union[str, Path],
        num_cores: int,
    ):
        utils.augment_data(
            midi_path=str(source_dir),
            augmented_dir=str(augmented_dir),
            augmented_tmp_dir=str(augmented_tmp_dir),
            num_cores=num_cores,
            data=self.name,
        )

    def encode_note_sequence(self, midi_path: Union[str, Path]) -> np.ndarray:
        # TODO
        # 1. 불필요한 io 작업 제거 (현재는 어쩔 수 없이 일단 이렇게 사용)
        # 2. 코드 트랙 제거시 좀더 우아한 방법 사용
        with tempfile.NamedTemporaryFile(suffix=Path(midi_path).suffix) as f:
            midi_obj = mido.MidiFile(midi_path)
            if len(midi_obj.tracks) > 2:
                midi_obj.tracks.pop(-1)
            midi_obj.save(f.name)
            return np.array(self.note_sequence_encoder.encode(f.name))

    @staticmethod
    def concat_npy(source_dir: Union[str, Path]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        def _gather(_prefix) -> List[str]:
            npy_suffix = ".npy"
            return sorted(
                str(f)
                for f in Path(source_dir).rglob("**/*")
                if f.suffix == npy_suffix and f.stem.startswith(_prefix)
            )

        def _concat(_npy_list: List[str]) -> List[np.ndarray]:
            return [np.load(_p, allow_pickle=True) for _p in _npy_list]

        return _concat(_gather("input")), _concat(_gather("target"))


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
        super().__init__(
            meta_parser=meta_parser,
            meta_encoder=meta_encoder,
            note_sequence_encoder=note_sequence_encoder,
            *args,
            **kwargs,
        )

    def preprocess(
        self,
        root_dir: Union[str, Path],
        num_cores: int,
        preprocess_steps: List[str],
        chunk_midi_arguments: ChunkMidiArguments,
        parse_midi_arguments: ParseMidiArguments,
        val_split_ratio: float = 0.1,
        test_split_ratio: float = 0.1,
        augment: bool = False,
    ) -> None:
        sub_dir = get_sub_dir(
            root_dir, exclude=None if augment else (SubDirName.AUGMENTED, SubDirName.AUGMENTED_TMP)
        )

        if augment:
            self.augment_data(
                source_dir=sub_dir.raw,
                augmented_dir=sub_dir.augmented,
                augmented_tmp_dir=sub_dir.encode_tmp,
                num_cores=num_cores,
            )

        if "chunk" in preprocess_steps:
            chunk_midi(
                midi_dataset_path=sub_dir.raw,
                chunked_midi_path=sub_dir.chunked,
                tmp_midi_dir=sub_dir.tmp,
                num_cores=num_cores,
                dataset_name=self.name,
                **asdict(chunk_midi_arguments),
            )

        if "parse" in preprocess_steps:
            for window_size in parse_midi_arguments.bar_window_size:
                utils.parse_midi(
                    source_dir=str(sub_dir.chunked),
                    num_measures=window_size,
                    shift_size=parse_midi_arguments.shift_size,
                    output_dir=sub_dir.parsed,
                    num_cores=num_cores,
                )
        self.export_encoded_midi(
            parsed_midi_dir=sub_dir.parsed,
            encode_tmp_dir=sub_dir.encode_tmp,
            num_cores=num_cores,
        )
        for npy_idx, sub_npy in enumerate(sub_dir.encode_tmp.iterdir()):
            inputs, targets = self.concat_npy(sub_npy)
            inputs = np.array([element for array in inputs for element in array])
            targets = np.array([element for array in targets for element in array])

            splits = utils.split_train_val_test(
                inputs,
                targets,
                val_ratio=val_split_ratio,
                test_ratio=test_split_ratio,
            )
            output_dir = sub_dir.encode_npy.joinpath(f"{npy_idx:04d}")
            output_dir.mkdir(exist_ok=True, parents=True)
            for split_name, data_split in splits.items():
                np.save(str(output_dir.joinpath(split_name)), data_split)

            del inputs, targets, splits
            gc.collect()

    def export_encoded_midi(
        self, parsed_midi_dir: Union[str, Path], encode_tmp_dir: Union[str, Path], num_cores: int
    ) -> None:
        for dir_idx, sub_pth in enumerate(parsed_midi_dir.iterdir()):
            midi_filenames = [
                str(filename)
                for filename in sub_pth.rglob("**/*")
                if filename.suffix in MIDI_EXTENSIONS
            ]
            _midi_filenames_chunk = np.array_split(np.array(midi_filenames), num_cores)
            # 프로세스별로 별도 idx 할당
            # TODO: `Manager().Lock`, `Manager().Value`를 사용하도록 수정할 것
            midi_filenames_chunk = [
                (idx, arr.tolist()) for idx, arr in enumerate(_midi_filenames_chunk)
            ]
            parmap.map(
                self._preprocess_midi_chunk,
                midi_filenames_chunk,
                encode_tmp_dir,
                dir_idx,
                pm_pbar=True,
                pm_processes=num_cores,
            )

    def _preprocess_midi_chunk(
        self,
        idx_midi_paths_chunk: Tuple[int, Iterable[Union[str, Path]]],
        encode_tmp_dir: Union[str, Path],
        dir_idx: int,
    ) -> None:
        idx, midi_paths_chunk = idx_midi_paths_chunk
        encode_tmp_dir = Path(encode_tmp_dir)
        res_note = []
        res_meta = []
        for _, midi_path in enumerate(midi_paths_chunk):
            try:
                encoding_output = self._preprocess_midi(midi_path)
                res_meta.append(encoding_output.meta)
                res_note.append(encoding_output.note_sequence)
            except UnprocessableMidiError:
                continue

        if res_note:
            output_dir = encode_tmp_dir.joinpath(f"{idx:04d}")
            output_dir.mkdir(exist_ok=True, parents=True)
            np.save(
                output_dir.joinpath(f"input_{dir_idx}"),
                np.array(res_meta),
            )
            np.save(
                output_dir.joinpath(f"target_{dir_idx}"),
                np.array(res_note),
            )

    def _preprocess_midi(self, midi_path: Union[str, Path]):
        encoded_meta: List[Union[int, str]] = self._encode_meta(self._parse_meta(midi_path))
        encoded_meta.append(constants.UNKNOWN)
        encoded_meta: np.ndarray = np.array(encoded_meta, dtype=object)
        encoded_note_sequence = np.array(self.encode_note_sequence(midi_path), dtype=object)
        return EncodingOutput(meta=encoded_meta, note_sequence=encoded_note_sequence)

    def _encode_meta(self, midi_meta: MidiMeta) -> List[int]:
        return self.meta_encoder.encode(midi_meta)

    def _parse_meta(self, midi_path: Union[str, Path]) -> MidiMeta:
        return self.meta_parser.parse(midi_path)


class Pozalabs2Preprocessor(RedditPreprocessor):
    name = "pozalabs2"

    def __init__(
        self,
        meta_parser: BaseMetaParser,
        meta_encoder: BaseMetaEncoder,
        note_sequence_encoder: MidiPerformanceEncoder,
        chord_progression_csv_path: Union[str, Path],
        *args,
        **kwargs,
    ):
        super().__init__(
            meta_parser=meta_parser,
            meta_encoder=meta_encoder,
            note_sequence_encoder=note_sequence_encoder,
            *args,
            **kwargs,
        )
        self.chord_progression_info = self._load_chord_progression_info(chord_progression_csv_path)

    def _preprocess_midi(self, midi_path: Union[str, Path]) -> Optional[EncodingOutput]:
        midi_meta = self.meta_parser.parse(midi_path, self.chord_progression_info)
        encoded_meta: List[Union[int, str]] = self._encode_meta(midi_meta)
        # TODO: 여러 코드 진행 처리 고려할 것
        midi_chord_progression_info = self.chord_progression_info.get(Path(midi_path).stem)
        chord_progression_md5 = (
            midi_chord_progression_info["chord_progression_hash"]
            if midi_chord_progression_info is not None
            else constants.UNKNOWN
        )
        encoded_meta.append(chord_progression_md5)
        encoded_meta: np.ndarray = np.array(encoded_meta, dtype=object)
        encoded_note_sequence = np.array(self.encode_note_sequence(midi_path), dtype=object)
        return EncodingOutput(meta=encoded_meta, note_sequence=encoded_note_sequence)

    @staticmethod
    def _load_chord_progression_info(
        chord_progression_csv_path: Union[str, Path]
    ) -> Dict[str, Dict[str, Any]]:
        df = pd.read_csv(chord_progression_csv_path)
        records = df.to_dict(orient="records")
        return {
            record["filename"]: {
                "chord_progression": record["chord_progression"].split(","),
                "chord_progression_hash": record["chord_progression_hash"],
            }
            for record in records
        }


class PozalabsPreprocessor(BasePreprocessor):
    name = "pozalabs"

    def __init__(
        self,
        meta_parser: BaseMetaParser,
        meta_encoder: BaseMetaEncoder,
        note_sequence_encoder: MidiPerformanceEncoder,
        backoffice_api_url: str,
        *args,
        **kwargs,
    ):
        super().__init__(
            meta_parser=meta_parser,
            meta_encoder=meta_encoder,
            note_sequence_encoder=note_sequence_encoder,
            *args,
            **kwargs,
        )
        self.backoffice_api_url = backoffice_api_url

    def preprocess(
        self,
        root_dir: Union[str, Path],
        num_cores: int,
        val_split_ratio: int = 0.1,
        test_split_ratio: int = 0.1,
        augment: bool = False,
    ):
        sub_dir_exclude = (SubDirName.CHUNKED, SubDirName.PARSED, SubDirName.TMP)
        if not augment:
            sub_dir_exclude = (*sub_dir_exclude, SubDirName.AUGMENTED, SubDirName.AUGMENTED_TMP)

        sub_dir = get_sub_dir(root_dir, exclude=sub_dir_exclude)
        fetched_samples = utils.load_poza_meta(
            self.backoffice_api_url + "/api/samples", per_page=2000
        )

        # 인터프리터가 if/else 모두 앞에 * (asterisk)가 붙어있는 것으로 해석하기 때문에 모든 인자에 튜플 사용
        sample_id_to_path = self._gather_sample_files(
            *(sub_dir.raw, sub_dir.augmented) if augment else (sub_dir.raw,)
        )

        if augment:
            self.augment_data(
                source_dir=sub_dir.raw,
                augmented_dir=sub_dir.augmented,
                augmented_tmp_dir=sub_dir.augmented_tmp,
                num_cores=num_cores,
            )

        self.export_encoded_midi(
            fetched_samples=fetched_samples,
            encoded_tmp_dir=sub_dir.encode_tmp,
            sample_id_to_path=sample_id_to_path,
            num_cores=num_cores,
        )
        splits = utils.split_train_val_test(
            *self.concat_npy(sub_dir.encode_tmp),
            val_ratio=val_split_ratio,
            test_ratio=test_split_ratio,
        )
        for split_name, data_split in splits.items():
            np.save(str(sub_dir.encode_npy.joinpath(split_name)), data_split)

    def export_encoded_midi(
        self,
        fetched_samples: List[Dict[str, Any]],
        sample_id_to_path: Dict[str, str],
        encoded_tmp_dir: Union[str, Path],
        num_cores: int,
    ) -> None:
        sample_infos_chunk = [
            (idx, arr.tolist())
            for idx, arr in enumerate(np.array_split(np.array(fetched_samples), num_cores))
        ]
        parmap.map(
            self._preprocess_midi_chunk,
            sample_infos_chunk,
            sample_id_to_path=sample_id_to_path,
            encode_tmp_dir=encoded_tmp_dir,
            pm_pbar=True,
            pm_processes=num_cores,
        )

    def _preprocess_midi_chunk(
        self,
        idx_sample_infos_chunk: Tuple[int, Iterable[Dict[str, Any]]],
        sample_id_to_path: Dict[str, str],
        encode_tmp_dir: Union[str, Path],
    ):
        idx, sample_infos_chunk = idx_sample_infos_chunk

        copied_sample_infos_chunk = copy.deepcopy(list(sample_infos_chunk))
        parent_sample_ids_to_info = {
            sample_info["id"]: sample_info for sample_info in copied_sample_infos_chunk
        }
        parent_sample_ids = set(parent_sample_ids_to_info.keys())

        copied_sample_infos_chunk.extend(
            [
                {"id": sample_id, "augmented": True}
                for sample_id in sample_id_to_path.keys()
                if sample_id.split("_")[0] in parent_sample_ids
            ]
        )

        encode_tmp_dir = Path(encode_tmp_dir)
        for sample_info_idx, sample_info in enumerate(copied_sample_infos_chunk):
            copied_sample_info = sample_info
            if sample_info.get("augmented", False):
                id_split = copied_sample_info["id"].split("_")

                bpm = copied_sample_info.get("bpm")
                audio_key = copied_sample_info.get("audio_key")
                if len(id_split) > 1:
                    parent_sample_id, audio_key, bpm = id_split
                else:
                    parent_sample_id = id_split[0]

                if bpm is None or audio_key is None:
                    continue

                copied_sample_info = copy.deepcopy(parent_sample_ids_to_info[parent_sample_id])
                copied_sample_info["id"] = copied_sample_info["id"]
                copied_sample_info["bpm"] = int(bpm)
                copied_sample_info["audio_key"] = audio_key

            if copied_sample_info["track_category"] in constants.NON_KEY_TRACK_CATEGORIES:
                continue

            if not copied_sample_info["chord_progressions"]:
                continue

            midi_path = sample_id_to_path.get(copied_sample_info["id"])
            # 백오피스에는 등록되었으나 아직 다운로드 되지 않은 샘플은 건너뜀
            if midi_path is None:
                continue

            encoding_output = self._preprocess_midi(
                sample_info=copied_sample_info, midi_path=midi_path
            )
            output_dir = encode_tmp_dir.joinpath(f"{idx:04d}")
            output_dir.mkdir(exist_ok=True, parents=True)
            np.save(output_dir.joinpath(f"input_{sample_info_idx}"), encoding_output.meta)
            np.save(output_dir.joinpath(f"target_{sample_info_idx}"), encoding_output.note_sequence)

    def _preprocess_midi(
        self, sample_info: Dict[str, Any], midi_path: Union[str, Path]
    ) -> EncodingOutput:
        midi_meta = self.meta_parser.parse(meta_dict=sample_info, midi_path=midi_path)
        encoded_meta: List[Union[int, str]] = self.meta_encoder.encode(midi_meta)
        # TODO: 여러 코드 진행 처리 고려할 것
        chord_progression_md5 = (
            utils.get_chord_progression_md5(midi_meta.chord_progression)
            if midi_meta.chord_progression != constants.UNKNOWN
            else constants.UNKNOWN
        )
        encoded_meta.append(chord_progression_md5)
        encoded_meta: np.ndarray = np.array(encoded_meta, dtype=object)
        encoded_note_sequence = np.array(self.encode_note_sequence(midi_path), dtype=object)
        return EncodingOutput(meta=encoded_meta, note_sequence=encoded_note_sequence)

    @staticmethod
    def _gather_sample_files(*source_dirs: Union[str, Path]) -> Dict[str, str]:
        def _gather(_source_dir):
            return {
                filename.stem: str(filename)
                for filename in Path(_source_dir).rglob("**/*")
                if filename.suffix in MIDI_EXTENSIONS
            }

        result = dict()
        for source_dir in source_dirs:
            result.update(_gather(source_dir))
        return result


PREPROCESSORS: Dict[str, Type[BasePreprocessor]] = {
    obj.name: obj
    for _, obj in globals().items()
    if inspect.isclass(obj) and issubclass(obj, BasePreprocessor) and not inspect.isabstract(obj)
}


class PreprocessorFactory:
    registered_preprocessors = tuple(PREPROCESSORS.keys())

    def create(self, name: str, **kwargs) -> BasePreprocessor:
        if name not in self.registered_preprocessors:
            raise ValueError(f"`name` should be one of {self.registered_preprocessors}")

        preprocessor_cls = PREPROCESSORS[name]
        return preprocessor_cls(**dependency.inject_args(preprocessor_cls.__init__, **kwargs))
