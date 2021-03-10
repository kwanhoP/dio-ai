import enum
import functools
import math
import tempfile
from pathlib import Path
from typing import List, Optional, Union

import mido
import numpy as np
import pretty_midi

from .exceptions import InvalidMidiError, InvalidMidiErrorMessage

DEFAULT_NUM_BEATS = 4
CHORD_TRACK_NAME = "chord"
PITCH_RANGE_CUT = {
    "very_low": 36,
    "low": 48,
    "mid_low": 60,
    "mid": 72,
    "mid_high": 84,
    "high": 96,
    "very_high": 108,
}
DEFAULT_PITCH_RANGE = "mid"
CHORD_TYPE_IDX = -1
MINOR_KEY = "m"
NO_META_MESSAGE = "no_info"
UNKNOWN = "unknown"


class ChordType(str, enum.Enum):
    MAJOR = "major"
    MINOR = "minor"

    @classmethod
    def values(cls) -> List[str]:
        return list(cls.__members__.values())


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
    time_signature: pretty_midi.TimeSignature = pt_midi.time_signature_changes[-1]

    coordination = time_signature.numerator / time_signature.denominator
    ticks_per_measure = pt_midi.resolution * DEFAULT_NUM_BEATS * coordination

    track = _get_track(pt_midi.instruments)
    if track is None:
        raise InvalidMidiError(InvalidMidiErrorMessage.chord_track_not_found.value)

    notes = track.notes

    # 노트가 시작하는 마디
    start_measure = pt_midi.time_to_tick(notes[0].start) // ticks_per_measure
    measure_start_tick = int(start_measure * ticks_per_measure)
    duration_tick = pt_midi.time_to_tick(notes[-1].end) - measure_start_tick
    return math.ceil(duration_tick / ticks_per_measure)


def get_pitch_range(midi_obj: mido.MidiFile, keyswitch_velocity: int) -> str:
    """미디의 피치 범위를 계산합니다. 메타/코드 트랙, 키스위치에 해당하는 노트는 계산에서 제외합니다.

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
        return total / count

    def _get_pitch_range(avg_pitch_range):
        indexer = {i: k for i, k in enumerate(PITCH_RANGE_CUT.keys())}
        bins = list(PITCH_RANGE_CUT.values())
        digitizer = functools.partial(np.digitize, bins=bins)
        return indexer[digitizer(avg_pitch_range)]

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
    """미디의 bpm을 추출합니다."""
    if isinstance(meta_message, str):
        return UNKNOWN
    return round(mido.tempo2bpm(getattr(meta_message, "tempo")))


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
