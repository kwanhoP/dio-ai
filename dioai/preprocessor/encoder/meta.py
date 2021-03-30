import abc
import enum
import functools
import inspect
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
}


class Unknown(int, enum.Enum):
    BPM = 0
    AUDIO_KEY = 41
    TIME_SIGNATURE = 66
    PITCH_RANGE = 84
    INST = 94


class Offset(int, enum.Enum):
    BPM = 1
    AUDIO_KEY = 42
    TIME_SIGNATURE = 67
    PITCH_RANGE = 85
    MEASURES_4 = 92
    MEASURES_8 = 93
    INST = 95


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
                return getattr(Unknown, meta_name.upper()).value
            return inject_args_to_encode_func(func, *args, **kwargs)

        return wrapper

    return decorator


def add_offset(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        meta_name = _get_meta_name(func.__name__).upper()
        offset_value = getattr(Offset, meta_name).value
        unknown_value = getattr(Unknown, meta_name).value
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
    return encoding_map[time_signature]


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
            ENCODERS[meta_name], getattr(midi_meta, meta_name), encoding_maps.get(meta_name)
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


META_ENCODERS: Dict[str, Type[BaseMetaEncoder]] = {
    obj.name: obj
    for _, obj in globals().items()
    if inspect.isclass(obj) and issubclass(obj, BaseMetaEncoder) and not inspect.isabstract(obj)
}
