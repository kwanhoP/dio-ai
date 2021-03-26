import functools
from pathlib import Path
from typing import Any, Dict, Optional, Union

import mido

from dioai.preprocessor import utils
from dioai.preprocessor.utils import constants
from dioai.preprocessor.utils.container import MidiMeta


class MidiMetaParser:
    def __init__(
        self,
        keyswitch_velocity: Optional[int] = None,
        default_pitch_range: str = "mid",
    ):
        self.keyswitch_velocity = keyswitch_velocity
        self.default_pitch_range = default_pitch_range

    def parse(
        self,
        midi_path: Optional[Union[str, Path]] = None,
        meta_dict: Optional[Dict[str, Any]] = None,
        default_to_unknown: bool = True,
    ) -> MidiMeta:
        if midi_path is None or meta_dict is None:
            raise ValueError("Either `midi_path` or `meta_dict` should not be None")

        if midi_path is not None:
            return self._parse(midi_path, default_to_unknown)
        return self._parse_from_dict(meta_dict)

    def _parse(self, midi_path: Union[str, Path], default_to_unknown: bool = True) -> MidiMeta:
        midi_obj = mido.MidiFile(midi_path)
        meta_track = midi_obj.tracks[0]

        _get_meta_message_func = functools.partial(utils.get_meta_message_v2, meta_track=meta_track)

        midi_meta = MidiMeta(
            bpm=utils.get_bpm_v2(_get_meta_message_func(event_type="set_tempo")),
            audio_key=utils.get_key(_get_meta_message_func(event_type="key_signature")),
            time_signature=utils.get_time_signature_v2(
                _get_meta_message_func(event_type="time_signature")
            ),
            pitch_range=utils.get_pitch_range_v2(
                midi_obj=midi_obj, keyswitch_velocity=self.keyswitch_velocity
            ),
            num_measures=utils.get_num_measures_from_midi(midi_obj.filename),
            inst=utils.get_inst_from_midi_v2(midi_obj.filename),
        )
        # reddit 데이터셋을 처리할 때 BPM/Key/Time signature 가 모두 기본값이면 UNKNOWN 처리
        if default_to_unknown and _is_all_default_meta(midi_meta):
            midi_meta = _all_default_to_unknown(midi_meta)
        return midi_meta

    @staticmethod
    def _parse_from_dict(meta_dict: Dict[str, Any]) -> MidiMeta:
        return MidiMeta(**meta_dict)


def _is_all_default_meta(midi_meta: MidiMeta) -> bool:
    return (
        midi_meta.bpm == constants.DefaultValue.BPM
        and midi_meta.audio_key == constants.DefaultValue.AUDIO_KEY
        and midi_meta.time_signature == constants.DefaultValue.TIME_SIGNATURE
    )


def _all_default_to_unknown(midi_meta: MidiMeta) -> MidiMeta:
    midi_meta.bpm = constants.UNKNOWN
    midi_meta.audio_key = constants.UNKNOWN
    midi_meta.time_signature = constants.UNKNOWN
    return midi_meta
