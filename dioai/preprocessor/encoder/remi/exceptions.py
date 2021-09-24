import enum


class REMIError(Exception):
    """remi encoder error"""


class InvalidMidiError(REMIError):
    """비정상 미디 입력 시 발생하는 에러"""


class InvalidMidiErrorMessage(str, enum.Enum):
    duplicate_meta = "미디 메타 메시지는 둘 이상 존재할 수 없습니다. (타입: {event_type})"


MIDI_EVENT_LABEL = {
    "time_signature": "Time Signature",
    "set_tempo": "Tempo",
    "key_signature": "Key Signature",
}
