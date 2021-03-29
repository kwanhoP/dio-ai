import copy
import enum
import functools
import http
import inspect
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mido
import numpy as np
import parmap
import pretty_midi
import requests
from sklearn.model_selection import train_test_split

from .constants import (
    BPM_INTERVAL,
    BPM_START_POINT,
    BPM_UNKHOWN,
    CHORD_TRACK_NAME,
    CHORD_TYPE_IDX,
    DEFAULT_NUM_BEATS,
    INST_PROGRAM_MAP,
    INST_START_POINT,
    INST_UNKHOWN,
    KEY_MAP,
    KEY_START_POINT,
    KEY_UNKHOWN,
    MAX_BPM,
    MEASURES_4,
    MEASURES_8,
    MINOR_KEY,
    NO_META_MESSAGE,
    PITCH_RANGE_CUT,
    PITCH_RANGE_MAP,
    PITCH_RANGE_START_POINT,
    PITCH_RANGE_UNKHOWN,
    PROGRAM_INST_MAP,
    SIG_TIME_MAP,
    TIME_SIG_MAP,
    TS_START_POINT,
    TS_UNKHOWN,
    UNKNOWN,
)
from .container import MidiInfo


class ChordType(str, enum.Enum):
    MAJOR = "major"
    MINOR = "minor"

    @classmethod
    def values(cls) -> List[str]:
        return list(cls.__members__.values())


def parse_midi(
    midi_path: str, num_measures: int, shift_size: int, parsing_midi_pth: Path, num_cores: int
) -> None:

    midifiles = []

    for _, (dirpath, _, filenames) in enumerate(os.walk(midi_path)):
        midi_extensions = [".mid", ".MID", ".MIDI", ".midi"]
        for ext in midi_extensions:
            tem = [os.path.join(dirpath, _) for _ in filenames if _.endswith(ext)]
            if tem:
                midifiles += tem

    split_midi = np.array_split(np.array(midifiles), num_cores)
    split_midi = [x.tolist() for x in split_midi]
    parmap.map(
        parse_midi_map,
        split_midi,
        num_measures,
        shift_size,
        parsing_midi_pth,
        pm_pbar=True,
        pm_processes=num_cores,
    )


def parse_midi_map(
    midifiles: List, num_measures: int, shift_size: int, parsing_midi_pth: Path
) -> None:
    for filename in midifiles:
        midi_data = pretty_midi.PrettyMIDI(filename)
        if len(midi_data.time_signature_changes) == 1:
            time_signature: pretty_midi.TimeSignature = midi_data.time_signature_changes[-1]
        elif len(midi_data.time_signature_changes) > 1:
            continue
        coordination = time_signature.numerator / time_signature.denominator
        ticks_per_measure = int(midi_data.resolution * DEFAULT_NUM_BEATS * coordination)
        midi_data.tick_to_time(ticks_per_measure)

        if not midi_data.instruments:
            continue
        notes = midi_data.instruments[0].notes
        parsing_duration = ticks_per_measure * num_measures
        shift_duration = ticks_per_measure * shift_size

        # Get Tempo
        _, tempo_infos = midi_data.get_tempo_changes()

        parsed_notes = []
        for start_tick in range(
            midi_data.time_to_tick(notes[0].start),
            int(midi_data.time_to_tick(midi_data.get_end_time()) - parsing_duration),
            int(shift_duration),
        ):
            end_tick = start_tick + parsing_duration
            tmp_note_list = []
            for i, note in enumerate(notes):
                new_note = copy.deepcopy(note)
                if (
                    start_tick <= midi_data.time_to_tick(new_note.start) < end_tick
                    or start_tick < midi_data.time_to_tick(new_note.end) <= end_tick
                ):
                    tmp_note_list.append(new_note)
            for i, note in enumerate(tmp_note_list):
                note.start = note.start - midi_data.tick_to_time(start_tick)
                note.end = note.end - midi_data.tick_to_time(start_tick)
                if note.start < 0.0:
                    note.start = 0.0
                if midi_data.time_to_tick(note.end) > parsing_duration:
                    note.end = midi_data.tick_to_time(parsing_duration)
            parsed_notes.append(tmp_note_list)

        for i, new_notes in enumerate(parsed_notes):
            new_midi_object = pretty_midi.PrettyMIDI(
                resolution=midi_data.resolution, initial_tempo=float(tempo_infos)
            )
            # key, 박자 입력
            ks_list = midi_data.key_signature_changes
            ts_list = midi_data.time_signature_changes

            if ks_list:  # ks 가 변화하지 않는 경우 default값으로 설정 필요
                for ks in ks_list:
                    new_midi_object.key_signature_changes.append(ks)

            if ts_list:  # ts 가 변화하지 않는 경우 default값으로 설정 필요
                for ts in ts_list:
                    new_midi_object.time_signature_changes.append(ts)

            # 노트 입력
            new_instrument = pretty_midi.Instrument(program=midi_data.instruments[0].program)
            new_instrument.notes = new_notes
            new_midi_object.instruments.append(new_instrument)
            filename_without_extension = os.path.splitext(filename.split("/")[-1])[0]
            new_midi_object.write(
                os.path.join(
                    parsing_midi_pth,
                    filename_without_extension + f"_{num_measures}_{i}.mid",
                )
            )


def get_inst_from_midi(midi_path: Union[str, Path]) -> int:
    """
    미디 program num을 poza 악기 분류로 mapping 하는 함수
    0: 건반악기, 1: 리드악기, 2: 체명악기, 3: 발현악기, 4: 현악기
    5: 금관악기, 6: 목관악기, 7: 기타, 8: 신스악기, 9: 타악기
    """
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    if not midi_data.instruments:
        return UNKNOWN
    else:
        program = midi_data.instruments[0].program
        return PROGRAM_INST_MAP[str(program)]


def get_inst_from_info(inst_type: int) -> int:
    return INST_PROGRAM_MAP[inst_type]


def get_ts_from_info(ts_type: int) -> [int, int]:
    numerator, denominator = SIG_TIME_MAP[ts_type].split("/")
    return int(numerator), int(denominator)


def get_num_measures_from_midi(
    midi_path: Union[str, Path], track_name: Optional[str] = None
) -> Union[int, str]:
    """미디에서 마디 수를 계산하는 함수"""

    def _get_track(tracks):
        if track_name is None:
            return tracks[0]
        for t in tracks:
            if t.name == track_name:
                return t

    pt_midi = pretty_midi.PrettyMIDI(str(midi_path))
    if not pt_midi.instruments:
        return UNKNOWN

    time_signature: pretty_midi.TimeSignature = pt_midi.time_signature_changes[-1]

    coordination = time_signature.numerator / time_signature.denominator
    ticks_per_measure = pt_midi.resolution * DEFAULT_NUM_BEATS * coordination

    track = _get_track(pt_midi.instruments)
    notes = track.notes

    # 노트가 시작하는 마디
    start_measure = pt_midi.time_to_tick(notes[0].start) // ticks_per_measure
    measure_start_tick = int(start_measure * ticks_per_measure)
    duration_tick = pt_midi.time_to_tick(notes[-1].end) - measure_start_tick
    return math.ceil(duration_tick / ticks_per_measure)


def get_pitch_range(midi_obj: mido.MidiFile, keyswitch_velocity: int) -> str:
    """미디의 피치 범위를 계산하고 인코딩합니다. 메타/코드 트랙, 키스위치에 해당하는 노트는 계산에서 제외합니다.

    Args:
        midi_obj: `mido.MidiFile`. mido.MidiFile 객체
        keyswitch_velocity: `int`. 키스위치로 설정한 노트의 velocity (default 1)
    """

    def _get_avg_note(_tracks: List[pretty_midi.Instrument]):
        total = 0
        count = 0
        for track in _tracks:
            # pretty_midi 에서 메타 트랙은 Instrument 로 파싱되지 않음
            if track.name == CHORD_TRACK_NAME:
                continue
            for event in track.notes:
                if event.pitch != 0 and event.velocity != keyswitch_velocity:
                    total += event.pitch
                    count += 1
        if count == 0:
            return UNKNOWN
        else:
            return total / count

    def _get_pitch_range(avg_pitch_range):
        if avg_pitch_range == UNKNOWN:
            return UNKNOWN
        else:
            indexer = {i: k for i, k in enumerate(PITCH_RANGE_CUT.keys())}
            bins = list(PITCH_RANGE_CUT.values())
            digitizer = functools.partial(np.digitize, bins=bins)
            try:
                range = indexer[digitizer(avg_pitch_range)]
            except KeyError:
                return UNKNOWN
        return PITCH_RANGE_MAP[range]

    with tempfile.NamedTemporaryFile(suffix=".mid") as f:
        midi_obj.save(filename=f.name)
        pt_midi = pretty_midi.PrettyMIDI(midi_file=f.name)

    return _get_pitch_range(_get_avg_note(pt_midi.instruments))


def get_time_signature(meta_message: mido.MetaMessage) -> Union[mido.MetaMessage, str]:
    """미디의 박자를 추출합니다."""
    if isinstance(meta_message, str):
        return UNKNOWN

    attrs = ("numerator", "denominator")
    time_sig = "/".join(str(getattr(meta_message, attr)) for attr in attrs)
    try:
        return TIME_SIG_MAP[time_sig]
    except KeyError:
        return UNKNOWN


def get_bpm(meta_message: mido.MetaMessage, poza_bpm: int) -> Union[mido.MetaMessage, str]:
    """미디의 bpm을 추출하여 BPM_INTERVAL(5단위)로 인코딩 합니다.
    poza_dataset의 경우 meta_message = None 으로 설정합니다."""
    if not poza_bpm:
        bpm = round(mido.tempo2bpm(getattr(meta_message, "tempo")))
    else:
        bpm = poza_bpm

    if isinstance(meta_message, str):
        return UNKNOWN

    if bpm >= MAX_BPM:
        return 39
    else:
        return bpm // BPM_INTERVAL


def get_key_chord_type(
    meta_message: mido.MetaMessage, lower: bool = True
) -> Union[mido.MetaMessage, str]:
    """미디의 key 정보(key, major/minor)를 추출합니다."""

    def _is_major(_ks):
        return _ks[CHORD_TYPE_IDX] != MINOR_KEY

    def _divide_key_chord_type(_ks, major):
        if major:
            return KEY_MAP[_ks + ChordType.MAJOR.value]
        return KEY_MAP[_ks[:CHORD_TYPE_IDX] + ChordType.MINOR.value]

    if isinstance(meta_message, str):
        return UNKNOWN

    else:
        key_signature: str = getattr(meta_message, "key")
        if lower:
            key_signature = key_signature.lower()

    return _divide_key_chord_type(key_signature, _is_major(key_signature))


def with_default(default_value: Any = UNKNOWN):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args):
            if len(args) > 1:
                raise ValueError("Only single argument is accepted")
            if args[0] is None:
                return default_value
            if not isinstance(args[0], mido.MetaMessage):
                arg_name = inspect.getfullargspec(func).args[0]
                raise TypeError(f"{arg_name} should be instance of `mido.MetaMessage`")
            return func(*args)

        return wrapper

    return decorator


@with_default()
def get_bpm_v2(meta_message: Optional[mido.MetaMessage]) -> Union[int, str]:
    return round(mido.tempo2bpm(getattr(meta_message, "tempo")))


@with_default()
def get_audio_key_v2(meta_message: Optional[mido.MetaMessage]) -> str:
    def _is_major(_ks):
        return _ks[CHORD_TYPE_IDX] != MINOR_KEY

    def _divide_key_chord_type(_ks, major):
        if major:
            return _ks, ChordType.MAJOR.value
        return _ks[:CHORD_TYPE_IDX], ChordType.MINOR.value

    key_signature = getattr(meta_message, "key")
    _key, _chord_type = _divide_key_chord_type(key_signature, _is_major(key_signature))
    return _key.lower() + _chord_type


@with_default()
def get_time_signature_v2(meta_message: Optional[mido.MetaMessage]) -> str:
    attrs = ("numerator", "denominator")
    time_signature = "/".join(str(getattr(meta_message, attr)) for attr in attrs)
    return time_signature


def get_pitch_range_v2(midi_obj: mido.MidiFile, keyswitch_velocity: Optional[int] = None) -> str:
    def _get_avg_note(_tracks: List[pretty_midi.Instrument]):
        total = 0
        count = 0
        for track in _tracks:
            # pretty_midi 에서 메타 트랙은 Instrument 로 파싱되지 않음
            if track.name == CHORD_TRACK_NAME:
                continue
            for event in track.notes:
                if event.pitch == 0:
                    continue

                if keyswitch_velocity is not None:
                    if event.velocity != keyswitch_velocity:
                        total += event.pitch
                        count += 1
                else:
                    total += event.pitch
                    count += 1
        if not count:
            return None
        return total / count

    def _get_pitch_range(avg_pitch_range):
        indexer = {i: k for i, k in enumerate(PITCH_RANGE_CUT.keys())}
        bins = list(PITCH_RANGE_CUT.values())
        digitizer = functools.partial(np.digitize, bins=bins)
        return indexer[digitizer(avg_pitch_range)]

    with tempfile.NamedTemporaryFile(suffix=".mid") as f:
        midi_obj.save(filename=f.name)
        pt_midi = pretty_midi.PrettyMIDI(midi_file=f.name)

    avg_note = _get_avg_note(pt_midi.instruments)
    if avg_note is None:
        return UNKNOWN
    return _get_pitch_range(avg_note)


def get_inst_from_midi_v2(midi_path: Union[str, Path]) -> Union[int, str]:
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    if not midi_data.instruments:
        return UNKNOWN
    return midi_data.instruments[0].program


def get_meta_message_v2(meta_track: mido.MidiTrack, event_type: str) -> Optional[mido.MetaMessage]:
    messages = [event for event in copy.deepcopy(meta_track) if event.type == event_type]
    if not messages:
        return None
    return messages.pop()


def get_meta_message(meta_track: mido.MidiTrack, event_type: str) -> Union[mido.MetaMessage, str]:
    """미도 파일을 mido.MetaMessage로 변환해 return 합니다."""

    messages = [event for event in meta_track if event.type == event_type]

    if not messages:
        return NO_META_MESSAGE

    return messages.pop()


def encode_meta_info(midi_info: MidiInfo) -> List:
    meta = []
    if midi_info.bpm is not UNKNOWN:
        meta.append(midi_info.bpm + BPM_START_POINT)
    else:
        meta.append(BPM_UNKHOWN)

    if midi_info.audio_key is not UNKNOWN:
        meta.append(midi_info.audio_key + KEY_START_POINT)
    else:
        meta.append(KEY_UNKHOWN)

    if midi_info.time_signature is not UNKNOWN:
        meta.append(midi_info.time_signature + TS_START_POINT)
    else:
        meta.append(TS_UNKHOWN)

    if midi_info.pitch_range is not UNKNOWN:
        meta.append(midi_info.pitch_range + PITCH_RANGE_START_POINT)
    else:
        meta.append(PITCH_RANGE_UNKHOWN)

    if midi_info.num_measure == 4:
        meta.append(MEASURES_4)
    elif midi_info.num_measure == 8:
        meta.append(MEASURES_8)
    else:
        return None

    if midi_info.inst is not UNKNOWN:
        meta.append(midi_info.inst + INST_START_POINT)
    else:
        meta.append(INST_UNKHOWN)

    return meta


def split_train_val_test(
    input: np.array, target: np.ndarray, test_ratio: float, val_ratio: float
) -> np.array:
    x_train, x_test, y_train, y_test = train_test_split(
        input, target, test_size=(test_ratio + val_ratio), shuffle=True, random_state=2021
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_test,
        y_test,
        test_size=(test_ratio / (test_ratio + val_ratio)),
        shuffle=True,
        random_state=2021,
    )
    splits = {
        "input_train": x_train,
        "input_val": x_val,
        "input_test": x_test,
        "target_train": y_train,
        "target_val": y_val,
        "target_test": y_test,
    }
    return splits


def load_poza_meta(request_url: str, per_page: int = 1000) -> List[Dict[str, Any]]:
    return fetch_samples_from_backoffice(
        request_url=request_url, per_page=per_page, params={"auto_changed": False}
    )


def fetch_samples_from_backoffice(
    request_url: str, per_page: int = 100, params: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    page = 1
    result = []
    finished = False
    while not finished:
        has_next, samples = _fetch_samples(request_url, page=page, per_page=per_page, params=params)
        result.extend(samples)
        finished = not has_next
        page += 1
    return result


def _fetch_samples(
    url, page: int = 1, per_page: int = 100, params: Optional[Dict[str, Any]] = None
) -> Tuple[bool, List[Dict[str, Any]]]:
    request_params = {"page": page, "per_page": per_page}
    if params is not None:
        request_params.update(params)

    res = requests.get(url, params=request_params)

    if res.status_code != http.HTTPStatus.OK:
        raise ValueError("Failed to fetch samples from backoffice")

    data = res.json()
    return data["has_next"], data["samples"]["samples"]


def concat_npy(encode_tmp_dir, MODEL, META_LEN):
    npy_list = os.listdir(encode_tmp_dir)

    input_npy_list = [
        os.path.join(encode_tmp_dir, npy_file)
        for npy_file in npy_list
        if npy_file.startswith("input")
    ]
    target_npy_list = [
        os.path.join(encode_tmp_dir, npy_file)
        for npy_file in npy_list
        if npy_file.startswith("target")
    ]

    input_lst = []
    target_lst = []
    for input_npy_pth in input_npy_list:
        _input_npy = np.load(input_npy_pth, allow_pickle=True)
        input_lst.append(_input_npy)

    for target_npy_pth in target_npy_list:
        _target_npy = np.load(target_npy_pth, allow_pickle=True)
        target_lst.append(_target_npy)

    input_npy = np.array(input_lst)
    target_npy = np.array(target_lst)

    if MODEL == "GPT":
        target_npy = target_npy + META_LEN
    return input_npy, target_npy
