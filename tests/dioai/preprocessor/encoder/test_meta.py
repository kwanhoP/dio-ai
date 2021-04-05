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
        meta.Unknown.get(name).value if name != "num_measures" else meta.Offset.MEASURES_4.value
        for name in meta.META_ENCODING_ORDER
    ]
    assert encoded_meta == expected


class TestPozalabsMetaEncoder:
    @pytest.fixture(scope="function", autouse=True)
    def _meta_encoder(self):
        self.meta_encoder = meta.PozalabsMetaEncoder()

    def test_encode(self):
        midi_meta = MidiMeta(
            bpm=200,
            audio_key="cmajor",
            time_signature="4/4",
            pitch_range="mid",
            num_measures="4",
            inst="acoustic_piano",
            genre="newage",
            min_velocity=12,
            max_velocity=127,
        )
        expected = [
            # 39 + Offset.BPM (423)
            462,
            # 0 + Offset.AUDIO_KEY (464)
            464,
            # 4 + Offset.TIME_SIGNATURE (489)
            493,
            # 3 + Offset.PITCH_RANGE (507)
            510,
            # Offset.MEASURES_4
            514,
            # 0 + Offset.INST (517)
            517,
            # 0 + Offset.GENRE (526)
            526,
            # 2 + Offset.VELOCITY
            542,
            # 26 + Offset.VELOCITY
            566,
        ]
        encoded_meta = self.meta_encoder.encode(midi_meta)
        assert encoded_meta == expected
