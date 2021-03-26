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

    # note
    note_seq: List[int]


class MidiMeta(BaseModel):
    bpm: Union[int, str]
    audio_key: str
    time_signature: str
    pitch_range: str
    num_measures: Union[int, str]
    inst: str
