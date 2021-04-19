import abc
import copy
import functools
import inspect
import re
from pathlib import Path
from typing import Any, Dict, Type, Union

import mido

from dioai.preprocessor import utils
from dioai.preprocessor.utils import constants
from dioai.preprocessor.utils.container import MidiMeta

from .utils import TableReader


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
        midi_path = str(midi_path)
        midi_obj = mido.MidiFile(midi_path)
        meta_track = midi_obj.tracks[0]

        _get_meta_message_func = functools.partial(utils.get_meta_message_v2, meta_track=meta_track)

        min_velocity, max_velocity = utils.get_velocity_range(midi_path)
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
            genre=utils.get_genre(midi_path.lower()),
            min_velocity=min_velocity,
            max_velocity=max_velocity,
            track_category=constants.UNKNOWN,
            chord_progression=constants.UNKNOWN,
        )
        # reddit 데이터셋을 처리할 때 BPM/Key/Time signature 가 모두 기본값이면 UNKNOWN 처리
        if self.default_to_unknown and _is_all_default_meta(midi_meta):
            midi_meta = _all_default_to_unknown(midi_meta)
        return midi_meta


class PozalabsMetaParser(BaseMetaParser):
    name = "pozalabs"

    def __init__(self):
        super().__init__()

    def parse(self, meta_dict: Dict[str, Any], midi_path: Union[str, Path]) -> MidiMeta:
        copied_meta_dict = copy.deepcopy(meta_dict)
        audio_key = copied_meta_dict["audio_key"]
        # 어그먼트된 데이터의 `audio_key`는 `KEY_MAP`에 정의된 형태로 기입되어 있음
        if not (utils.ChordType.MAJOR in audio_key or utils.ChordType.MINOR in audio_key):
            copied_meta_dict["audio_key"] = (
                copied_meta_dict["audio_key"] + copied_meta_dict["chord_type"]
            )
        copied_meta_dict["inst"] = remove_number_from_inst(copied_meta_dict["inst"])
        copied_meta_dict["chord_progression"] = copied_meta_dict.pop("chord_progressions")[0]
        min_velocity, max_velocity = utils.get_velocity_range(
            midi_path,
            keyswitch_velocity=constants.KeySwitchVelocity.get_value(copied_meta_dict["inst"]),
        )
        return MidiMeta(**copied_meta_dict, min_velocity=min_velocity, max_velocity=max_velocity)


class Pozalabs2MetaParser(BaseMetaParser):
    name = "pozalabs2"

    def __init__(self, meta_csv_path: Union[str, Path]):
        super().__init__()
        self.genre_info = TableReader(meta_csv_path).get_meta_dict()

    def parse(
        self, midi_path: Union[str, Path], chord_progression_info: Dict[str, Dict[str, Any]]
    ) -> MidiMeta:
        midi_name = Path(midi_path).stem
        midi_chord_progression_info = chord_progression_info.get(midi_name)

        midi_path = str(midi_path)
        midi_obj = mido.MidiFile(midi_path)
        meta_track = midi_obj.tracks[0]

        _get_meta_message_func = functools.partial(utils.get_meta_message_v2, meta_track=meta_track)

        min_velocity, max_velocity = utils.get_velocity_range(midi_path)
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
            genre=utils.get_genre(midi_path, self.genre_info),
            min_velocity=min_velocity,
            max_velocity=max_velocity,
            # 코드 트랙은 항상 마지막 트랙
            track_category=utils.get_track_category_from_channel(midi_obj.tracks[1]),
            chord_progression=(
                midi_chord_progression_info["chord_progression"]
                if midi_chord_progression_info is not None
                else constants.UNKNOWN
            ),
        )
        return midi_meta


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


def remove_number_from_inst(inst: str) -> str:
    """포자랩스 샘플 정보에 기입된 악기에서 숫자를 제거합니다. 악기명 형식은 `{inst}-[0-9]`입니다."""
    inst_number_pattern = re.compile("-[0-9]+")
    return inst_number_pattern.sub("", inst)
