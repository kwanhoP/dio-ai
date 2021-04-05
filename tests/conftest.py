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
    )
