import os
import tempfile
from typing import Union

import mido
import numpy as np
import pytest

from dioai.preprocessor import chunk_midi


@pytest.fixture(scope="function")
def midi_path_fixture():
    fd, path = tempfile.mkstemp(suffix=".mid")
    os.close(fd)
    midi_obj = mido.MidiFile()
    track = mido.MidiTrack()
    track.extend(
        [
            mido.MetaMessage("set_tempo", tempo=1000000, time=0),
            mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0),
            mido.MetaMessage("key_signature", key="C", time=0),
        ]
    )
    track = mido.MidiTrack()
    track.extend(
        [
            mido.MetaMessage("track_name", name="main_melody"),
            mido.Message("note_on", channel=0, note=58, velocity=1, time=0),
            mido.Message("note_on", channel=0, note=74, velocity=60, time=0),
            mido.Message("note_off", channel=0, note=58, velocity=1, time=440),
            mido.Message("note_off", channel=0, note=74, velocity=60, time=440),
        ]
    )
    midi_obj.tracks.append(track)
    midi_obj.save(path)
    yield path
    os.remove(path)


def test_apply_channel(midi_path_fixture):
    channel = 4
    chunk_midi.apply_channel(midi_path_fixture, track_to_channel={"main_melody": channel})
    midi_obj = mido.MidiFile(midi_path_fixture)
    for track in midi_obj.tracks:
        for event in track:
            if hasattr(event, "channel"):
                assert event.channel == channel


@pytest.mark.parametrize(
    "f, n, dtype, expected",
    [
        (0.1, 9, "float", 0.1),
        (0.1, 3, "str", "0.100"),
        (1e-2, 3, "float", 0.01),
        (1e-2, 3, "str", "0.010"),
    ],
)
def test_truncate(f: float, n: int, dtype: str, expected: Union[str, float]):
    assert chunk_midi.truncate(f, n, dtype) == expected


@pytest.mark.parametrize(
    "event_times, tempo_infos, end_time, expected",
    [
        (np.array([0.0]), np.array([120.0]), 60.0, 120),
        (np.array([0.0, 60.0]), np.array([100.0, 120.0]), 80.0, 105),
    ],
)
def test_get_avg_bpm(
    event_times: np.ndarray, tempo_infos: np.ndarray, end_time: float, expected: int
):
    result = chunk_midi.get_avg_bpm(
        event_times=event_times, tempo_infos=tempo_infos, end_time=end_time
    )
    assert result == expected
