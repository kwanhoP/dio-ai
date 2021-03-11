import enum
import functools
import math
import tempfile
from pathlib import Path
from typing import List, Optional, Union

import mido
import numpy as np
import pretty_midi

from .constants import (
    CHORD_TRACK_NAME,
    CHORD_TYPE_IDX,
    DEFAULT_NUM_BEATS,
    DEFAULT_PITCH_RANGE,
    MINOR_KEY,
    NO_META_MESSAGE,
    PITCH_RANGE_CUT,
    PITCH_RANGE_MAP,
    PROGRAM_INST_MAP,
    UNKNOWN,
)
from .exceptions import InvalidMidiError, InvalidMidiErrorMessage


class ChordType(str, enum.Enum):
    MAJOR = "major"
    MINOR = "minor"

    @classmethod
    def values(cls) -> List[str]:
        return list(cls.__members__.values())


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
        pro_num = midi_data.instruments[0].program
        return PROGRAM_INST_MAP[str(pro_num)]


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
    return "/".join(str(getattr(meta_message, attr)) for attr in attrs)


def get_bpm(meta_message: mido.MetaMessage) -> Union[mido.MetaMessage, str]:
    """미디의 bpm을 추출하여 5단위로 인코딩 합니다."""
    bpm = round(mido.tempo2bpm(getattr(meta_message, "tempo")))

    if isinstance(meta_message, str):
        return UNKNOWN

    if bpm == 200:
        return 39
    else:
        return bpm // 5


def get_key_chord_type(
    meta_message: mido.MetaMessage, lower: bool = True
) -> Union[mido.MetaMessage, str]:
    """미디의 key 정보(key, major/minor)를 추출합니다."""

    def _is_major(_ks):
        return _ks[CHORD_TYPE_IDX] != MINOR_KEY

    def _divide_key_chord_type(_ks, major):
        if major:
            return _ks, ChordType.MAJOR.value
        return _ks[:CHORD_TYPE_IDX], ChordType.MINOR.value

    if isinstance(meta_message, str):
        key_signature = UNKNOWN

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
