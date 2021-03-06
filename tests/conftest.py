import pytest

from dioai.preprocessor.utils import constants
from dioai.preprocessor.utils.container import MidiMeta


@pytest.fixture(scope="function")
def unknown_meta() -> MidiMeta:
    return MidiMeta(
        bpm=constants.UNKNOWN,
        audio_key=constants.UNKNOWN,
        time_signature=constants.UNKNOWN,
        pitch_range=constants.UNKNOWN,
        num_measures=constants.UNKNOWN,
        inst=constants.UNKNOWN,
        genre=constants.UNKNOWN,
        min_velocity=constants.UNKNOWN,
        max_velocity=constants.UNKNOWN,
        track_category=constants.UNKNOWN,
        chord_progression=constants.UNKNOWN,
        rhythm=constants.UNKNOWN,
        min_modulation=constants.UNKNOWN,
        max_modulation=constants.UNKNOWN,
        min_expression=constants.UNKNOWN,
        max_expression=constants.UNKNOWN,
        min_sustain=constants.UNKNOWN,
        max_sustain=constants.UNKNOWN,
    )
