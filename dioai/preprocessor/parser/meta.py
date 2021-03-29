import abc
import functools
import inspect
from pathlib import Path
from typing import Any, Dict, Type, Union

import mido

from dioai.preprocessor import utils
from dioai.preprocessor.utils import constants
from dioai.preprocessor.utils.container import MidiMeta


class BaseMetaParser(abc.ABC):
    name = ""

    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def parse(self, *args, **kwargs) -> MidiMeta:
        raise NotImplementedError


class RedditMetaParser(BaseMetaParser):
    name = "reddit"

    def __init__(self, default_to_unknown: bool = True):
        super().__init__()
        self.default_to_unknown = default_to_unknown

    def parse(self, midi_path: Union[str, Path]) -> MidiMeta:
        midi_obj = mido.MidiFile(midi_path)
        meta_track = midi_obj.tracks[0]

        _get_meta_message_func = functools.partial(utils.get_meta_message_v2, meta_track=meta_track)

        midi_meta = MidiMeta(
            bpm=utils.get_bpm_v2(_get_meta_message_func(event_type="set_tempo")),
            audio_key=utils.get_audio_key_v2(_get_meta_message_func(event_type="key_signature")),
            time_signature=utils.get_time_signature_v2(
                _get_meta_message_func(event_type="time_signature")
            ),
            pitch_range=utils.get_pitch_range_v2(
                midi_obj=midi_obj, keyswitch_velocity=constants.KeySwitchVelocity.DEFAULT
            ),
            num_measures=utils.get_num_measures_from_midi(midi_obj.filename),
            inst=utils.get_inst_from_midi_v2(midi_obj.filename),
        )
        # reddit 데이터셋을 처리할 때 BPM/Key/Time signature 가 모두 기본값이면 UNKNOWN 처리
        if self.default_to_unknown and _is_all_default_meta(midi_meta):
            midi_meta = _all_default_to_unknown(midi_meta)
        return midi_meta


class PozalabsMetaParser(BaseMetaParser):
    name = "pozalabs"

    def __init__(self):
        super().__init__()

    def parse(self, meta_dict: Dict[str, Any]) -> MidiMeta:
        # TODO: 포자랩스의 `inst`는 실제 악기명이지 프로그램 번호가 아니므로 제대로 파싱하도록 수정해야 함
        return MidiMeta(**meta_dict)


META_PARSERS: Dict[str, Type[BaseMetaParser]] = {
    obj.name: obj
    for _, obj in globals().items()
    if inspect.isclass(obj) and issubclass(obj, BaseMetaParser) and not inspect.isabstract(obj)
}


def _is_all_default_meta(midi_meta: MidiMeta) -> bool:
    return (
        midi_meta.bpm == constants.DefaultValue.BPM.value
        and midi_meta.audio_key == constants.DefaultValue.AUDIO_KEY.value
        and midi_meta.time_signature == constants.DefaultValue.TIME_SIGNATURE.value
    )


def _all_default_to_unknown(midi_meta: MidiMeta) -> MidiMeta:
    midi_meta.bpm = constants.UNKNOWN
    midi_meta.audio_key = constants.UNKNOWN
    midi_meta.time_signature = constants.UNKNOWN
    return midi_meta
