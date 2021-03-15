from dataclasses import dataclass
from typing import List


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
