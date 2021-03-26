import pytest

from dioai.exceptions import ErrorMessage, UnprocessableMidiError
from dioai.preprocessor.encoder import meta
from dioai.preprocessor.utils.container import MidiMeta


def test_encode_meta_all_unknown(unknown_meta: MidiMeta):
    with pytest.raises(UnprocessableMidiError) as exc_info:
        meta.encode_meta(unknown_meta)

        assert ErrorMessage.UNPROCESSABLE_MIDI_ERROR.value in str(exc_info.value)


def test_encode_meta_unknown_encoded(unknown_meta: MidiMeta):
    unknown_meta.num_measures = 4
    encoded_meta = meta.encode_meta(unknown_meta)
    expected = [
        (
            getattr(meta.Unknown, name.upper()).value
            if name != "num_measures"
            else meta.Offset.MEASURES_4.value
        )
        for name in meta.META_ENCODING_ORDER
    ]
    assert encoded_meta == expected


def test_encode():
    midi_meta = MidiMeta(
        bpm=200,
        audio_key="cmajor",
        time_signature="4/4",
        pitch_range="mid",
        num_measures="4",
        inst="0",
    )
    expected = [
        # 39 + Offset.BPM (1)
        40,
        # 0 + Offset.AUDIO_KEY (42)
        42,
        # 4 + Offset.TIME_SIGNATURE (67)
        71,
        # 3 + Offset.PITCH_RANGE (85)
        88,
        # Offset.MEASURES_4
        92,
        # 0 + Offset.INST (95)
        95,
    ]
    encoded_meta = meta.encode_meta(midi_meta)
    assert encoded_meta == expected
