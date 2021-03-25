import enum


class InvalidMidiErrorMessage(str, enum.Enum):
    duplicate_meta = "미디 메타 메시지는 둘 이상 존재할 수 없습니다. (타입: {event_type})"
    invalid_chord = "올바르지 않은 코드입니다. (관련 코드: {}) 혹은 코드 노트의 길이를 확인하세요."
    invalid_chord_progression_length = "코드 트랙의 마디 수는 2 또는 4의 배수여야 합니다."
    chord_track_not_found = "코드 트랙을 찾을 수 없습니다."
    invalid_chord_note_length = "올바르지 않은 코드 음표입니다. 8분 음표 단위 (0.5)의 음표만 사용할 수 있습니다."
    inconsistent_chord_progression_length = "모든 코드 트랙의 마디 수는 동일해야 합니다."


class InvalidMidiError(Exception):
    """비정상 미디 입력 시 발생하는 에러"""
