from dataclasses import dataclass

import mido

from .utils import (
    get_bpm,
    get_key_chord_type,
    get_meta_message,
    get_num_measures_from_midi,
    get_pitch_range,
    get_time_signature,
)


@dataclass
class MidiMeta:
    bpm: int
    audio_key: str
    chord_type: str
    time_signature: str
    pitch_range: str
    num_measure: int
    inst: int


class MetaExtractor:
    """미디 메타 정보를 추출합니다.

    파싱되는 정보:
        - bpm
        - audio_key
        - chord_type
        - time_signature
        - pitch_range
        - num_measure
    """

    def __init__(
        self,
        pth: str,
        keyswitch_velocity: int,
        default_pitch_range: str,
    ):
        """

        Args:
            midi: `str`. 파싱할 미디 path
            keyswitch_velocity: `int`. pitch range 검사에서 제외할 keyswitch velocity
            default_pitch_range: `str`. 모든 노트의 velocity 가 keyswitch velocity 라서
                        pitch range를 검사할 수 없을 경우 사용할 기본 pitch range

        """
        self._midi = mido.MidiFile(pth)
        self.keyswitch_velocity = keyswitch_velocity
        self.default_pitch_range = default_pitch_range
        self.path = pth

    def parse(self) -> MidiMeta:
        meta_track = self._midi.tracks[0]
        audio_key, chord_type = get_key_chord_type(get_meta_message(meta_track, "key_signature"))

        return MidiMeta(
            bpm=get_bpm(get_meta_message(meta_track, "set_tempo")),
            audio_key=audio_key,
            chord_type=chord_type,
            time_signature=get_time_signature(get_meta_message(meta_track, "time_signature")),
            pitch_range=get_pitch_range(self._midi, self.keyswitch_velocity),
            num_measure=get_num_measures_from_midi(self.path),
        )
