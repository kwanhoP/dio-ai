import abc
import enum
import functools
import inspect
import math
from typing import Any, Callable, Dict, List, Type, Union

from dioai.exceptions import ErrorMessage, UnprocessableMidiError
from dioai.preprocessor.utils import constants
from dioai.preprocessor.utils.container import MidiMeta

EncodeFunc = Union[Callable[[Any], int], Callable[[Any, Dict[Any, int]], int]]
# !!WARNING!!
# `pydantic.BaseModel.__fields__`에는 필드를 선언한 순서대로 필드 정보가 기입되므로,
# `MidiMeta`의 필드 선언 순서에 주의해야 합니다.
META_ENCODING_ORDER = tuple(MidiMeta.__fields__.keys())
DEFAULT_ENCODING_MAPS = {
    "audio_key": constants.KEY_MAP,
    "time_signature": constants.TIME_SIG_MAP,
    "pitch_range": constants.PITCH_RANGE_MAP,
    "inst": constants.PROGRAM_INST_MAP,
    "genre": constants.GENRE_MAP,
    "track_category": constants.TRACK_CATEGORY_MAP,
    "rhythm": constants.RHYTHM_MAP,
}
ATTR_ALIAS = {
    "min_velocity": "velocity",
    "max_velocity": "velocity",
    "min_modulation": "modulation",
    "max_modulation": "modulation",
    "min_expression": "expression",
    "max_expression": "expression",
    "min_sustain": "sustain",
    "max_sustain": "sustain",
}
META_TO_ENCODER_ALIAS = {"chord_progression": "has_chord_progression"}


class AliasMixin:
    @classmethod
    def get(cls, key: str):
        key = key.lower()
        if key in ATTR_ALIAS:
            return getattr(cls, ATTR_ALIAS[key].upper())
        return getattr(cls, key.upper())


class Unknown(AliasMixin, int, enum.Enum):
    BPM = 422
    AUDIO_KEY = 463
    TIME_SIGNATURE = 488
    PITCH_RANGE = 506
    INST = 516
    GENRE = 525
    VELOCITY = 539
    TRACK_CATEGORY = 567
    RHYTHM = 576
    MODULATION = 579
    EXPRESSION = 607
    SUSTAIN = 635


class Offset(AliasMixin, int, enum.Enum):
    BPM = 423
    AUDIO_KEY = 464
    TIME_SIGNATURE = 489
    PITCH_RANGE = 507
    MEASURES_4 = 514
    MEASURES_8 = 515
    INST = 517
    GENRE = 526
    VELOCITY = 540
    TRACK_CATEGORY = 568
    HAS_CHORD_PROGRESSION = 574
    RHYTHM = 577
    MODULATION = 580
    EXPRESSION = 608
    SUSTAIN = 636


ENCODERS: Dict[str, EncodeFunc] = dict()


def _get_meta_name(func_name: str) -> str:
    return "_".join(func_name.split("_")[1:])


def register_encoder(func):
    ENCODERS[_get_meta_name(func.__name__)] = func
    return func


def inject_args_to_encode_func(encode_func, *args, **kwargs) -> int:
    num_args = len(inspect.getfullargspec(encode_func).args)
    if num_args == 1:
        return encode_func(args[0])
    return encode_func(*args, **kwargs)


def encode_unknown(
    raise_error: bool = False, error_message: str = ErrorMessage.UNPROCESSABLE_MIDI_ERROR.value
):
    def decorator(func: EncodeFunc):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            meta_name = _get_meta_name(func.__name__)
            if args[0] == constants.UNKNOWN:
                if raise_error:
                    raise UnprocessableMidiError(error_message)
                return Unknown.get(meta_name).value
            return inject_args_to_encode_func(func, *args, **kwargs)

        return wrapper

    return decorator


def add_offset(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        meta_name = _get_meta_name(func.__name__).upper()
        offset_value = Offset.get(meta_name).value
        unknown_value = Unknown.get(meta_name).value
        result = inject_args_to_encode_func(func, *args, **kwargs)
        if result == unknown_value:
            return result
        return result + offset_value

    return wrapper


@register_encoder
@add_offset
@encode_unknown()
def encode_bpm(bpm: Union[int, str]) -> int:
    # 인코딩 결괏값이 40이면 Offset을 더했을 때 AUDIO_KEY Unknown으로 잘못 인코딩 됨
    return min(bpm, constants.MAX_BPM - 1) // constants.BPM_INTERVAL


@register_encoder
@add_offset
@encode_unknown()
def encode_audio_key(audio_key: str, encoding_map: Dict[str, int]) -> int:
    return encoding_map[audio_key]


@register_encoder
@add_offset
@encode_unknown()
def encode_time_signature(time_signature: str, encoding_map: Dict[str, int]) -> int:
    try:
        return encoding_map[time_signature]
    except KeyError:
        raise UnprocessableMidiError(f"ts keyerror: {time_signature}")


@register_encoder
@add_offset
@encode_unknown()
def encode_pitch_range(pitch_range: str, encoding_map: Dict[str, int]) -> int:
    return encoding_map[pitch_range]


@register_encoder
@encode_unknown(raise_error=True)
def encode_num_measures(num_measures: Union[int, str]) -> int:
    if num_measures == 4:
        return Offset.MEASURES_4.value
    return Offset.MEASURES_8.value


@register_encoder
@add_offset
@encode_unknown()
def encode_inst(inst: Union[int, str], encoding_map: Dict[str, int]) -> int:
    return encoding_map[inst]


@register_encoder
@add_offset
@encode_unknown()
def encode_genre(genre: str, encoding_map: Dict[str, int]) -> int:
    return encoding_map[genre]


@register_encoder
@add_offset
@encode_unknown()
def encode_min_velocity(velocity: Union[int, str]):
    # 몫을 사용하지 않는 이유 관련 주석: https://github.com/POZAlabs/dio-ai/pull/73/files#r606947174
    return math.floor(velocity / constants.VELOCITY_INTERVAL)


@register_encoder
@add_offset
@encode_unknown()
def encode_max_velocity(velocity: Union[int, str]):
    return math.ceil(velocity / constants.VELOCITY_INTERVAL)


@register_encoder
@add_offset
@encode_unknown()
def encode_track_category(track_category: str, encoding_map: Dict[str, int]) -> int:
    return encoding_map[track_category]


@register_encoder
def encode_has_chord_progression(chord_progression: Union[str, List[str]]) -> int:
    if chord_progression == constants.UNKNOWN:
        return Offset.HAS_CHORD_PROGRESSION.value + 1
    return Offset.HAS_CHORD_PROGRESSION.value


@register_encoder
@add_offset
@encode_unknown()
def encode_rhythm(rhythm: str, encoding_map: Dict[str, int]) -> int:
    return encoding_map[rhythm]


@register_encoder
@add_offset
@encode_unknown()
def encode_min_modulation(modulation: Union[int, str]):
    return math.floor(modulation / constants.MODULATION_INTERVAL)


@register_encoder
@add_offset
@encode_unknown()
def encode_max_modulation(modulation: Union[int, str]):
    return math.ceil(modulation / constants.MODULATION_INTERVAL)


@register_encoder
@add_offset
@encode_unknown()
def encode_min_expression(expression: Union[int, str]):
    return math.floor(expression / constants.EXPRESSION_INTERVAL)


@register_encoder
@add_offset
@encode_unknown()
def encode_max_expression(expression: Union[int, str]):
    return math.ceil(expression / constants.EXPRESSION_INTERVAL)


@register_encoder
@add_offset
@encode_unknown()
def encode_min_sustain(sustain: Union[int, str]):
    return math.floor(sustain / constants.SUSTAIN_INTERVAL)


@register_encoder
@add_offset
@encode_unknown()
def encode_max_sustain(sustain: Union[int, str]):
    return math.ceil(sustain / constants.SUSTAIN_INTERVAL)


def encode_meta(
    midi_meta: MidiMeta,
    encoding_maps_override: Dict[Any, int] = None,
) -> List[int]:
    encoding_maps = (
        {**DEFAULT_ENCODING_MAPS, **encoding_maps_override}
        if encoding_maps_override is not None
        else DEFAULT_ENCODING_MAPS
    )
    result = []
    for meta_name in META_ENCODING_ORDER:
        encoded_meta = inject_args_to_encode_func(
            ENCODERS[META_TO_ENCODER_ALIAS.get(meta_name, meta_name)],
            getattr(midi_meta, meta_name),
            encoding_maps.get(meta_name),
        )
        result.append(encoded_meta)
    return result


# 굳이 추상클래스로 선언하지 않아도 되지만 `MetaParser`와의 통일성을 위해 추상클래스로 선언
class BaseMetaEncoder(abc.ABC):
    name = ""
    encoding_maps_override = dict()

    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def encode(self, midi_meta: MidiMeta) -> List[int]:
        raise NotImplementedError


class RedditMetaEncoder(BaseMetaEncoder):
    name = "reddit"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode(self, midi_meta: MidiMeta) -> List[int]:
        return encode_meta(midi_meta)


class PozalabsMetaEncoder(BaseMetaEncoder):
    name = "pozalabs"
    encoding_maps_override = {"inst": constants.POZA_INST_MAP}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode(self, midi_meta: MidiMeta) -> List[int]:
        return encode_meta(midi_meta, encoding_maps_override=self.encoding_maps_override)


class Pozalabs2MetaEncoder(BaseMetaEncoder):
    name = "pozalabs2"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode(self, midi_meta: MidiMeta) -> List[int]:
        return encode_meta(midi_meta)


META_ENCODERS: Dict[str, Type[BaseMetaEncoder]] = {
    obj.name: obj
    for _, obj in globals().items()
    if inspect.isclass(obj) and issubclass(obj, BaseMetaEncoder) and not inspect.isabstract(obj)
}
