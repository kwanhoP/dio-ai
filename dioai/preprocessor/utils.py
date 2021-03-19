import copy
import enum
import functools
import http
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mido
import numpy as np
import pretty_midi
import requests
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .constants import (
    BPM_INTERVAL,
    BPM_START_POINT,
    BPM_UNKHOWN,
    CHORD_TRACK_NAME,
    CHORD_TYPE_IDX,
    DEFAULT_NUM_BEATS,
    DEFAULT_PITCH_RANGE,
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
from .exceptions import InvalidMidiError, InvalidMidiErrorMessage


class ChordType(str, enum.Enum):
    MAJOR = "major"
    MINOR = "minor"

    @classmethod
    def values(cls) -> List[str]:
        return list(cls.__members__.values())


def parse_midi(midi_path: str, num_measures: int, shift_size: int, parsing_midi_pth: Path) -> None:

    midifiles = []

    for i, (dirpath, _, filenames) in enumerate(os.walk(midi_path)):
        fileExt = [".mid", ".MID", ".MIDI", ".midi"]
        for Ext in fileExt:
            tem = [os.path.join(dirpath, _) for _ in filenames if _.endswith(Ext)]
            if tem:
                midifiles += tem

    for filename in tqdm(midifiles):
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
) -> int:
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
    else:
        time_signature: pretty_midi.TimeSignature = pt_midi.time_signature_changes[-1]

        coordination = time_signature.numerator / time_signature.denominator
        ticks_per_measure = pt_midi.resolution * DEFAULT_NUM_BEATS * coordination

        track = _get_track(pt_midi.instruments)
        if track is []:
            raise InvalidMidiError(InvalidMidiErrorMessage.chord_track_not_found.value)

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
            range = DEFAULT_PITCH_RANGE
        else:
            indexer = {i: k for i, k in enumerate(PITCH_RANGE_CUT.keys())}
            bins = list(PITCH_RANGE_CUT.values())
            digitizer = functools.partial(np.digitize, bins=bins)
            range = indexer[digitizer(avg_pitch_range)]
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
    return TIME_SIG_MAP[time_sig]


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


def load_poza_meta(url, page: int = 1, per_page: int = 100) -> List[Dict[str, Any]]:
    res = requests.get(url, params={"page": page, "per_page": per_page, "auto_changed": False})

    if res.status_code != http.HTTPStatus.OK:
        raise ValueError("Failed to fetch samples from backoffice")

    data = res.json()
    return data["samples"]["samples"]
