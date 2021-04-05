import os
import tempfile
from typing import Dict, Optional, Tuple, Union

import mido
import pretty_midi
import pytest

from dioai.preprocessor import utils
from dioai.preprocessor.utils import constants


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
            mido.MetaMessage("track_name", name="main_melody", time=0),
            mido.Message("note_on", channel=0, note=58, velocity=1, time=0),
            mido.Message("note_on", channel=0, note=74, velocity=60, time=0),
            mido.Message("note_off", channel=0, note=58, velocity=1, time=440),
            mido.Message("note_off", channel=0, note=74, velocity=60, time=440),
        ]
    )
    track.name = "main_melody"
    print(list(track))
    midi_obj.tracks.append(track)
    midi_obj.save(path)
    yield path
    os.remove(path)


@pytest.fixture(scope="function")
def midi_obj_fixture():
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
            mido.MetaMessage("track_name", name="main_melody", time=0),
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
        ("midi_path_fixture", "0"),
        ("midi_path_fixture_no_instrument", constants.UNKNOWN),
    ],
)
def test_get_inst_from_midi_v2(midi_path: str, expected: str, request: pytest.FixtureRequest):
    midi_path = request.getfixturevalue(midi_path)
    assert utils.get_inst_from_midi_v2(midi_path) == expected


@pytest.mark.parametrize(
    "midi_path, path_to_genre, expected",
    [
        ("BG-000005-c_1_0_0_8_5.mid", {"BG-000005-c": "newage"}, "newage"),
        ("Jazz_www.thejazzpage.de_MIDIRip/whisper.mid", None, "jazz"),
        ("Lyric-Pieces-Opus-12-Nr-1.mid", None, "cinematic"),
    ],
)
def test_get_genre(midi_path: str, path_to_genre: Optional[Dict[str, str]], expected: str):
    assert utils.get_genre(midi_path=midi_path, path_to_genre=path_to_genre) == expected


@pytest.mark.parametrize(
    "midi_path, keyswitch_velocity, expected",
    [
        ("midi_path_fixture", 1, (60, 60)),
        ("midi_path_fixture", None, (1, 60)),
        ("midi_path_fixture_no_instrument", None, (constants.UNKNOWN, constants.UNKNOWN)),
    ],
)
def test_get_velocity_range(
    midi_path: str,
    keyswitch_velocity: Optional[int],
    expected: Tuple[Union[int, str], Union[int, str]],
    request: pytest.FixtureRequest,
):
    midi_path = request.getfixturevalue(midi_path)
    result = utils.get_velocity_range(midi_path=midi_path, keyswitch_velocity=keyswitch_velocity)
    assert result == expected


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


def test_apply_channel(midi_path_fixture):
    channel = 4
    utils.apply_channel(midi_path_fixture, track_to_channel={"main_melody": channel})
    midi_obj = mido.MidiFile(midi_path_fixture)
    for track in midi_obj.tracks:
        for event in track:
            if hasattr(event, "channel"):
                assert event.channel == channel
