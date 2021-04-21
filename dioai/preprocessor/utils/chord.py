import tempfile
from dataclasses import dataclass, field
from typing import List, Optional

import mido
import music21

from dioai.preprocessor.utils import constants, utils
from dioai.preprocessor.utils.exceptions import InvalidMidiError, InvalidMidiErrorMessage


@dataclass
class MidiChordProgressionInfo:
    chord_progressions: Optional[List[List[str]]] = field(default=None)
    quantized: bool = field(default=True)


class ChordParser:
    """미디 파일에서 코드를 파싱합니다. 단위 음인 8분음표로 파싱합니다. 샘플 자동 생성 과정에서 생성되는 미디는
    메타 메시지 (lyrics)에서 코드 정보를 가져오며, 그렇지 않을 경우 코드 트랙에서 직접 코드를 파싱합니다.
    """

    def __init__(self, midi: mido.MidiFile, auto_changed: bool = False):
        """
        Args:
            midi: `mido.MidiFile`. 파싱할 미디 객체
            auto_changed: `bool`. 부모 샘플 여부. 부모 샘플인 경우는 다음과 같습니다.
                1) 미디/웨이브/패치를 직접 등록한 경우
                2) 자동 생성되었지만 BPM/Key가 변환되지 않은 경우
        """
        self._midi = midi
        self.auto_changed = auto_changed

    def parse(
        self, unit_beat: float = constants.EIGHTH_NOTE_BEATS, *, allow_various_lengths: bool = False
    ) -> MidiChordProgressionInfo:
        """코드를 단위 음표인 8분음표로 파싱합니다.
        Args:
            unit_beat: `float`. 단위 음표의 길이 (기본값 0.5 (8분음표))
            allow_various_lengths: `bool`. 이 값이 `True`이면 다양한 길이의 코드 트랙 허용
                (주의: 해당 인자는 코드 샘플을 파싱할 때에만 사용해야 함)
        Returns:
            `MidiChordProgressionInfo`
        """

        def _check_quantized(_chord_stream: music21.stream.Stream) -> bool:
            original_offsets = [e.offset for e in _chord_stream]
            quantized_offsets = [e.offset for e in _chord_stream.quantize()]
            return original_offsets == quantized_offsets

        if not self._check_chord_progression_length_even():
            raise InvalidMidiError(InvalidMidiErrorMessage.invalid_chord_progression_length.value)
        with tempfile.NamedTemporaryFile(suffix=".mid") as f:
            self._midi.save(filename=f.name)
            chord_streams = utils.get_chord_streams(f.name)
        quantized = all(_check_quantized(chord_stream) for chord_stream in chord_streams)

        chord_progressions = []
        for chord_stream in chord_streams:
            chord_progression = self._parse(chord_stream, unit_beat)

            if not self._check_chord_progression_length_even(num_measures=len(chord_progression)):
                raise InvalidMidiError(
                    InvalidMidiErrorMessage.invalid_chord_progression_length.value
                )
            chord_progressions.append(chord_progression)

        if not allow_various_lengths:
            chord_progression_lengths = set(len(c) for c in chord_progressions)
            if len(chord_progression_lengths) > 1:
                raise InvalidMidiError(
                    InvalidMidiErrorMessage.inconsistent_chord_progression_length.value
                )

        return MidiChordProgressionInfo(chord_progressions=chord_progressions, quantized=quantized)

    # 8분음표를 기본 단위로 사용 (파싱을 테스트하기 위한 `unit_beat` 파라미터 설정)
    @staticmethod
    def _parse(chord_stream: music21.stream.Stream, unit_beat: float) -> List[str]:
        chord_progression = []
        for idx, event in enumerate(chord_stream, 1):
            if not isinstance(event, music21.chord.Chord):
                continue

            if not utils.check_divisible_by_unit_beat(event.quarterLength, unit_beat):
                additional_info = f" (위치: {idx}, 길이: {event.duration.quarterLength})"
                message = InvalidMidiErrorMessage.invalid_chord_note_length.value + additional_info
                raise InvalidMidiError(message)

            chord = utils.find_chord_name(event.notes)
            chords = utils.divide_chord_into_unit_beat(chord, event.quarterLength, unit_beat)
            chord_progression.extend(chords)
        return chord_progression

    def _check_chord_progression_length_even(self, num_measures: Optional[int] = None) -> bool:
        """허용되는 마디 길이인지 확인합니다. 2 혹은 4의 배수만 허용됩니다.
        Args:
            num_measures: `Optional[int]`. 코드 트랙의 마디 수. 입력하지 않을 경우 마디 수를 계산합니다.
        Returns:
            `bool`. 허용되는 마디 길이인지 여부
        """
        if num_measures is None:
            with tempfile.NamedTemporaryFile(suffix=".mid") as f:
                self._midi.save(filename=f.name)
                try:
                    num_measures = utils.get_num_measures_from_midi_v2(f.name, "chord")
                except InvalidMidiError:
                    raise
        return (
            num_measures == constants.UNIT_MEASURES
            or num_measures % (constants.UNIT_MEASURES ** 2) == 0
        )
