import enum


class ErrorMessage(str, enum.Enum):
    UNPROCESSABLE_MIDI_ERROR = "Unprocessable midi"


class DioaiError(Exception):
    pass


class UnprocessableMidiError(DioaiError):
    """인코딩할 수 없는 미디"""
