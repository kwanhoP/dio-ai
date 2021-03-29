import os
import tempfile
from typing import Optional, Union

import mido
import pretty_midi
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


@pytest.fixture(scope="function")
def midi_path_fixture():
    fd, filepath = tempfile.mkstemp(suffix=".mid")
    os.close(fd)
    midi_obj = pretty_midi.PrettyMIDI()
    track = pretty_midi.Instrument(program=0)
    track.notes.extend(
        [
            pretty_midi.Note(velocity=60, pitch=60, start=0.0, end=2.0),
            pretty_midi.Note(velocity=60, pitch=72, start=0.0, end=4.0),
        ]
    )
    midi_obj.instruments.append(track)
    midi_obj.write(filepath)
    yield filepath
    os.remove(filepath)


@pytest.fixture(scope="function")
def midi_path_fixture_no_instrument():
    fd, filepath = tempfile.mkstemp(suffix=".mid")
    os.close(fd)
    midi_obj = pretty_midi.PrettyMIDI()
    midi_obj.write(filepath)
    yield filepath
    os.remove(filepath)


@pytest.mark.parametrize(
    "meta_message, expected",
    [
        (mido.MetaMessage("set_tempo", tempo=500000, time=0), 120),
        (None, constants.UNKNOWN),
    ],
)
def test_get_bpm_v2(meta_message: Optional[mido.MetaMessage], expected: str):
    assert utils.get_bpm_v2(meta_message) == expected


@pytest.mark.parametrize(
    "meta_message, expected",
    [
        (mido.MetaMessage("key_signature", key="C", time=0), "cmajor"),
        (None, constants.UNKNOWN),
    ],
)
def test_get_audio_key_v2(meta_message: Optional[mido.MetaMessage], expected: str):
    assert utils.get_audio_key_v2(meta_message) == expected


@pytest.mark.parametrize(
    "meta_message, expected",
    [
        (mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0), "4/4"),
        (None, constants.UNKNOWN),
    ],
)
def test_get_time_signature_v2(meta_message: Optional[mido.MetaMessage], expected: str):
    assert utils.get_time_signature_v2(meta_message) == expected


@pytest.mark.parametrize(
    "midi_obj, keyswitch_velocity, expected",
    [
        ("midi_obj_fixture", 1, "mid_high"),
        ("midi_obj_fixture", None, "mid"),
        ("midi_obj_fixture_no_message", None, constants.UNKNOWN),
    ],
)
def test_get_pitch_range_v2(
    midi_obj: str,
    keyswitch_velocity: Optional[int],
    expected: str,
    request: pytest.FixtureRequest,
):
    midi_obj = request.getfixturevalue(midi_obj)
    result = utils.get_pitch_range_v2(midi_obj=midi_obj, keyswitch_velocity=keyswitch_velocity)
    assert result == expected


@pytest.mark.parametrize(
    "midi_path, expected",
    [
        ("midi_path_fixture", 0),
        ("midi_path_fixture_no_instrument", constants.UNKNOWN),
    ],
)
def test_get_inst_from_midi_v2(
    midi_path: str, expected: Union[int, str], request: pytest.FixtureRequest
):
    midi_path = request.getfixturevalue(midi_path)
    assert utils.get_inst_from_midi_v2(midi_path) == expected


@pytest.mark.parametrize(
    "meta_track, event_type, expected",
    [
        (
            mido.MidiTrack(
                (mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0),)
            ),
            "time_signature",
            mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0),
        ),
        (
            mido.MidiTrack(),
            "time_signature",
            None,
        ),
    ],
)
def test_get_meta_message_v2(
    meta_track: mido.MidiTrack, event_type: str, expected: Optional[mido.MetaMessage]
):
    result = utils.get_meta_message_v2(meta_track=meta_track, event_type=event_type)
    assert result == expected
