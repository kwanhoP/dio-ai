from typing import Optional

import mido
import pytest

from dioai.preprocessor import utils
from dioai.preprocessor.utils import constants


@pytest.fixture(scope="function")
def midi_obj_fixture():
    midi_obj = mido.MidiFile()
    track = mido.MidiTrack()
    track.extend(
        [
            mido.Message("note_on", channel=0, note=58, velocity=1, time=0),
            mido.Message("note_on", channel=0, note=74, velocity=60, time=0),
            mido.Message("note_off", channel=0, note=58, velocity=1, time=440),
            mido.Message("note_off", channel=0, note=74, velocity=60, time=440),
        ]
    )
    midi_obj.tracks.append(track)
    return midi_obj


@pytest.fixture(scope="function")
def midi_obj_fixture_no_message():
    midi_obj = mido.MidiFile()
    track = mido.MidiTrack()
    midi_obj.tracks.append(track)
    return midi_obj


@pytest.mark.parametrize(
    "midi_obj, keyswitch_velocity, expected",
    [
        ("midi_obj_fixture", 1, "mid_high"),
        ("midi_obj_fixture", None, "mid"),
        ("midi_obj_fixture_no_message", None, constants.UNKNOWN),
    ],
)
def test_get_pitch_range_v2(
    midi_obj: mido.MidiFile,
    keyswitch_velocity: Optional[int],
    expected: str,
    request: pytest.FixtureRequest,
):
    midi_obj = request.getfixturevalue(midi_obj)
    result = utils.get_pitch_range_v2(midi_obj=midi_obj, keyswitch_velocity=keyswitch_velocity)
    assert result == expected
