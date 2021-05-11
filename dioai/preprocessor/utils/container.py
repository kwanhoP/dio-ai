from dataclasses import dataclass
from typing import List, Union

from pydantic import BaseModel


@dataclass
class MidiInfo:
    # meta
    bpm: int
    audio_key: int
    time_signature: int
    pitch_range: int
    num_measure: int
    inst: int
    genre: str
    min_velocity: int
    max_velocity: int
    track_category: str
    chord_progression: List[str]
    rhythm: int
    min_modulation: int
    max_modulation: int
    min_expression: int
    max_expression: int
    min_sustain: int
    max_sustain: int

    # note
    note_seq: List[int]


class MidiMeta(BaseModel):
    bpm: Union[int, str]
    audio_key: str
    time_signature: str
    pitch_range: str
    num_measures: Union[int, str]
    inst: str
    genre: str
    min_velocity: Union[int, str]
    max_velocity: Union[int, str]
    track_category: str
    chord_progression: Union[str, List[str]]
    rhythm: str
    min_modulation: Union[int, str]
    max_modulation: Union[int, str]
    min_expression: Union[int, str]
    max_expression: Union[int, str]
    min_sustain: Union[int, str]
    max_sustain: Union[int, str]
