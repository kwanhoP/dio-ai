import copy
import enum
import functools
import hashlib
import http
import inspect
import math
import os
import re
import tempfile
from fractions import Fraction
from multiprocessing import Lock, Pool, Value
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mido
import music21
import music21.midi.translate
import numpy as np
import parmap
import pretty_midi
import requests
import tqdm
from sklearn.model_selection import train_test_split

from dioai.exceptions import UnprocessableMidiError
from dioai.preprocessor.utils.exceptions import InvalidMidiError, InvalidMidiErrorMessage

from . import constants
from .constants import (
    BPM_INTERVAL,
    BPM_START_POINT,
    BPM_UNKHOWN,
    CHORD_TRACK_NAME,
    CHORD_TYPE_IDX,
    DEFAULT_NUM_BEATS,
    INST_PROGRAM_MAP,
    INST_START_POINT,
    INST_UNKHOWN,
    KEY_MAP,
    KEY_START_POINT,
    KEY_UNKHOWN,
    MAX_BPM,
    MEASURES_4,
    MEASURES_8,
    MINOR_KEY,
    MINOR_KEY_OFFSET,
    NO_META_MESSAGE,
    NUM_BPM_AUGMENT,
    NUM_KEY_AUGMENT,
    PITCH_RANGE_CUT,
    PITCH_RANGE_MAP,
    PITCH_RANGE_START_POINT,
    PITCH_RANGE_UNKHOWN,
    PRETTY_MAJOR_KEY,
    PRETTY_MINOR_KEY,
    PROGRAM_INST_MAP,
    SIG_TIME_MAP,
    TIME_SIG_MAP,
    TS_START_POINT,
    TS_UNKHOWN,
    UNKNOWN,
)
from .container import MidiInfo


def get_chord_progression_md5(chord_progression: List[str]) -> str:
    chord_progression_str = ",".join(chord_progression)
    return hashlib.md5(chord_progression_str.encode()).hexdigest()


class Counter:
    def __init__(self):
        self._value = Value("i", 0)
        self._lock = Lock()

    @property
    def value(self) -> int:
        with self._lock:
            return self._value.value

    def increment(self):
        with self._lock:
            self._value.value += 1


class ChordType(str, enum.Enum):
    MAJOR = "major"
    MINOR = "minor"

    @classmethod
    def values(cls) -> List[str]:
        return list(cls.__members__.values())


_counter = Counter()


def parse_midi(
    source_dir: str,
    num_cores: int,
    num_measures: int,
    shift_size: int,
    output_dir: Union[str, Path],
    max_keep: int = 20000,
    preserve_channel: bool = False,
    chunk_size: int = 200,
) -> None:
    midi_paths = [
        str(filename)
        for filename in Path(source_dir).rglob("**/*")
        if filename.suffix in constants.MIDI_EXTENSIONS
    ]

    parse_midi_map_partial = functools.partial(
        parse_midi_map,
        num_measures=num_measures,
        shift_size=shift_size,
        output_dir=output_dir,
        max_keep=max_keep,
        preserve_channel=preserve_channel,
    )

    with Pool(num_cores) as pool:
        with tqdm.tqdm(total=len(midi_paths), desc="Parsing") as pbar:
            for _ in pool.imap_unordered(parse_midi_map_partial, midi_paths, chunksize=chunk_size):
                pbar.update()


def parse_midi_map(
    midi_path: Union[str, Path],
    num_measures: int,
    shift_size: int,
    output_dir: Union[str, Path],
    max_keep: int = 20000,
    preserve_channel: bool = False,
) -> None:
    def _get_channel_info(_path):
        _raw_channel_info = {
            _track.name: get_channel(_track) for _track in mido.MidiFile(_path).tracks
        }
        _channel_info = {
            track_name: channel
            for track_name, channel in _raw_channel_info.items()
            if channel is not None
        }
        return _channel_info

    def _get_parsing_range(
        _pt_midi: pretty_midi.PrettyMIDI, _notes: List[pretty_midi.Note], _has_chord_track: bool
    ) -> Tuple[float]:
        if _has_chord_track:
            chord_notes = _pt_midi.instruments[1].notes
            is_incomplete_measure = chord_notes[0].start > 0
            if is_incomplete_measure:
                note_start_sec = chord_notes[0].start
                note_end_sec = chord_notes[-1].end
                return (note_start_sec, note_end_sec)
        note_start_sec = _notes[0].start
        note_end_sec = _pt_midi.get_end_time()
        return (note_start_sec, note_end_sec)

    def _get_note_segments(
        _pt_midi: pretty_midi.PrettyMIDI,
        _notes: List[pretty_midi.Note],
        _start_tick: int,
        _end_tick: int,
    ) -> List[pretty_midi.Note]:
        _new_notes = []
        for _note in _notes:
            _new_note = copy.deepcopy(_note)
            if (
                _start_tick <= _pt_midi.time_to_tick(_new_note.start) < _end_tick
                or _start_tick < _pt_midi.time_to_tick(_new_note.end) <= _end_tick
            ):
                if _new_note.start < _pt_midi.tick_to_time(_start_tick):
                    _new_note.start = _pt_midi.tick_to_time(_start_tick)
                if _new_note.end > _pt_midi.tick_to_time(_end_tick):
                    _new_note.end = _pt_midi.tick_to_time(_end_tick)
                _new_notes.append(_new_note)
        return _new_notes

    def _shift_start_time(
        _pt_midi: pretty_midi.PrettyMIDI, _notes: List[pretty_midi.Note], _start_tick: int
    ) -> List[pretty_midi.Note]:
        for _note in _notes:
            _note.start -= _pt_midi.tick_to_time(_start_tick)
            _note.end -= _pt_midi.tick_to_time(_start_tick)
        return _notes

    filename = midi_path

    output_dir = Path(output_dir).joinpath(f"{_counter.value:04d}")
    output_dir.mkdir(parents=True, exist_ok=True)

    num_processed = len(list(output_dir.glob("*.mid")))
    if num_processed > 0 and num_processed > max_keep:
        _counter.increment()

    track_to_channel = None
    if preserve_channel:
        track_to_channel = _get_channel_info(filename)
    midi_data = pretty_midi.PrettyMIDI(filename)

    time_signature_changes = midi_data.time_signature_changes
    if len(time_signature_changes) > 1:
        return None

    time_signature = time_signature_changes[-1]
    coordination = time_signature.numerator / time_signature.denominator
    ticks_per_measure = int(midi_data.resolution * DEFAULT_NUM_BEATS * coordination)

    if not midi_data.instruments:
        return None

    has_chord_track = len(midi_data.instruments) > 1
    notes = midi_data.instruments[0].notes
    parsing_duration = ticks_per_measure * num_measures
    shift_duration = ticks_per_measure * shift_size

    # Get Tempo
    _, tempo_infos = midi_data.get_tempo_changes()

    note_start_sec, note_end_sec = _get_parsing_range(midi_data, notes, has_chord_track)
    parsed_notes: List[List[pretty_midi.Note]] = []
    parsed_chord_notes: List[List[pretty_midi.Note]] = []
    for start_tick in range(
        midi_data.time_to_tick(note_start_sec),
        int(midi_data.time_to_tick(note_end_sec) - parsing_duration + shift_duration),
        int(shift_duration),
    ):
        end_tick = start_tick + parsing_duration
        new_notes = _get_note_segments(midi_data, notes, start_tick, end_tick)
        new_notes = _shift_start_time(midi_data, new_notes, start_tick)
        parsed_notes.append(new_notes)

        if has_chord_track:
            chord_notes = midi_data.instruments[1].notes
            chord_notes = _get_note_segments(midi_data, chord_notes, start_tick, end_tick)
            chord_notes = _shift_start_time(midi_data, chord_notes, start_tick)
            parsed_chord_notes.append(chord_notes)

    if not all(len(notes) for notes in parsed_notes):
        return

    for i, new_notes in enumerate(parsed_notes):
        if has_chord_track:
            chord_notes = parsed_chord_notes[i]
            if not chord_notes:
                continue
            else:
                if min([note.start for note in chord_notes]) > midi_data.tick_to_time(0):
                    continue
                if max([note.end for note in chord_notes]) < midi_data.tick_to_time(
                    parsing_duration
                ):
                    continue

        new_midi_object = pretty_midi.PrettyMIDI(
            resolution=midi_data.resolution, initial_tempo=float(tempo_infos)
        )
        # key, ?????? ??????
        ks_list = midi_data.key_signature_changes
        ts_list = midi_data.time_signature_changes

        if ks_list:  # ks ??? ???????????? ?????? ?????? default????????? ?????? ??????
            for ks in ks_list:
                new_midi_object.key_signature_changes.append(ks)

        if ts_list:  # ts ??? ???????????? ?????? ?????? default????????? ?????? ??????
            for ts in ts_list:
                new_midi_object.time_signature_changes.append(ts)

        # ?????? ??????
        new_instrument = pretty_midi.Instrument(
            program=midi_data.instruments[0].program, name=midi_data.instruments[0].name
        )
        new_instrument.notes = new_notes
        new_midi_object.instruments.append(new_instrument)
        if has_chord_track:
            new_chord_track = pretty_midi.Instrument(
                program=midi_data.instruments[1].program,
                name=midi_data.instruments[1].name,
            )
            new_chord_track.notes = chord_notes
            new_midi_object.instruments.append(new_chord_track)

        output_path = str(output_dir.joinpath(f"{Path(filename).stem}_{num_measures}_{i}.mid"))
        new_midi_object.write(output_path)
        if preserve_channel:
            apply_channel(output_path, track_to_channel=track_to_channel)


def get_inst_from_midi(midi_path: Union[str, Path]) -> int:
    """
    ?????? program num??? poza ?????? ????????? mapping ?????? ??????
    0: ????????????, 1: ????????????, 2: ????????????, 3: ????????????, 4: ?????????
    5: ????????????, 6: ????????????, 7: ??????, 8: ????????????, 9: ?????????
    """
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    if not midi_data.instruments:
        return UNKNOWN
    else:
        program = midi_data.instruments[0].program
        return PROGRAM_INST_MAP[str(program)]


def get_inst_from_info(inst_type: int) -> int:
    return INST_PROGRAM_MAP[inst_type]


def get_ts_from_info(ts_type: int) -> [int, int]:
    numerator, denominator = SIG_TIME_MAP[ts_type].split("/")
    return int(numerator), int(denominator)


def get_num_measures_from_midi(
    midi_path: Union[str, Path], track_name: Optional[str] = None
) -> Union[int, str]:
    """???????????? ?????? ?????? ???????????? ??????"""

    def _get_track(tracks):
        if track_name is None:
            return tracks[0]
        for t in tracks:
            if t.name == track_name:
                return t

    pt_midi = pretty_midi.PrettyMIDI(str(midi_path))
    if not pt_midi.instruments:
        return UNKNOWN

    time_signature: pretty_midi.TimeSignature = pt_midi.time_signature_changes[-1]

    coordination = time_signature.numerator / time_signature.denominator
    ticks_per_measure = pt_midi.resolution * DEFAULT_NUM_BEATS * coordination

    track = _get_track(pt_midi.instruments)
    notes = track.notes

    # ????????? ???????????? ??????
    start_measure = pt_midi.time_to_tick(notes[0].start) // ticks_per_measure
    measure_start_tick = int(start_measure * ticks_per_measure)
    duration_tick = pt_midi.time_to_tick(notes[-1].end) - measure_start_tick
    return math.ceil(duration_tick / ticks_per_measure)


def get_pitch_range(midi_obj: mido.MidiFile, keyswitch_velocity: int) -> str:
    """????????? ?????? ????????? ???????????? ??????????????????. ??????/?????? ??????, ??????????????? ???????????? ????????? ???????????? ???????????????.

    Args:
        midi_obj: `mido.MidiFile`. mido.MidiFile ??????
        keyswitch_velocity: `int`. ??????????????? ????????? ????????? velocity (default 1)
    """

    def _get_avg_note(_tracks: List[pretty_midi.Instrument]):
        total = 0
        count = 0
        for track in _tracks:
            # pretty_midi ?????? ?????? ????????? Instrument ??? ???????????? ??????
            if track.name == CHORD_TRACK_NAME:
                continue
            for event in track.notes:
                if event.pitch != 0 and event.velocity != keyswitch_velocity:
                    total += event.pitch
                    count += 1
        if count == 0:
            return UNKNOWN
        else:
            return total / count

    def _get_pitch_range(avg_pitch_range):
        if avg_pitch_range == UNKNOWN:
            return UNKNOWN
        else:
            indexer = {i: k for i, k in enumerate(PITCH_RANGE_CUT.keys())}
            bins = list(PITCH_RANGE_CUT.values())
            digitizer = functools.partial(np.digitize, bins=bins)
            try:
                range = indexer[digitizer(avg_pitch_range)]
            except KeyError:
                return UNKNOWN
        return PITCH_RANGE_MAP[range]

    with tempfile.NamedTemporaryFile(suffix=".mid") as f:
        midi_obj.save(filename=f.name)
        pt_midi = pretty_midi.PrettyMIDI(midi_file=f.name)

    return _get_pitch_range(_get_avg_note(pt_midi.instruments))


def get_time_signature(meta_message: mido.MetaMessage) -> Union[mido.MetaMessage, str]:
    """????????? ????????? ???????????????."""
    if isinstance(meta_message, str):
        return UNKNOWN

    attrs = ("numerator", "denominator")
    time_sig = "/".join(str(getattr(meta_message, attr)) for attr in attrs)
    try:
        return TIME_SIG_MAP[time_sig]
    except KeyError:
        return UNKNOWN


def get_bpm(meta_message: mido.MetaMessage, poza_bpm: int) -> Union[mido.MetaMessage, str]:
    """????????? bpm??? ???????????? BPM_INTERVAL(5??????)??? ????????? ?????????.
    poza_dataset??? ?????? meta_message = None ?????? ???????????????."""
    if not poza_bpm:
        bpm = round(mido.tempo2bpm(getattr(meta_message, "tempo")))
    else:
        bpm = poza_bpm

    if isinstance(meta_message, str):
        return UNKNOWN

    if bpm >= MAX_BPM:
        return 39
    else:
        return bpm // BPM_INTERVAL


def get_key_chord_type(
    meta_message: mido.MetaMessage, lower: bool = True
) -> Union[mido.MetaMessage, str]:
    """????????? key ??????(key, major/minor)??? ???????????????."""

    def _is_major(_ks):
        return _ks[CHORD_TYPE_IDX] != MINOR_KEY

    def _divide_key_chord_type(_ks, major):
        if major:
            return KEY_MAP[_ks + ChordType.MAJOR.value]
        return KEY_MAP[_ks[:CHORD_TYPE_IDX] + ChordType.MINOR.value]

    if isinstance(meta_message, str):
        return UNKNOWN

    else:
        key_signature: str = getattr(meta_message, "key")
        if lower:
            key_signature = key_signature.lower()

    return _divide_key_chord_type(key_signature, _is_major(key_signature))


def with_default(default_value: Any = UNKNOWN):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args):
            if len(args) > 1:
                raise ValueError("Only single argument is accepted")
            if args[0] is None:
                return default_value
            if not isinstance(args[0], mido.MetaMessage):
                arg_name = inspect.getfullargspec(func).args[0]
                raise TypeError(f"{arg_name} should be instance of `mido.MetaMessage`")
            return func(*args)

        return wrapper

    return decorator


@with_default()
def get_bpm_v2(meta_message: Optional[mido.MetaMessage]) -> Union[int, str]:
    return round(mido.tempo2bpm(getattr(meta_message, "tempo")))


@with_default()
def get_audio_key_v2(meta_message: Optional[mido.MetaMessage]) -> str:
    def _is_major(_ks):
        return _ks[CHORD_TYPE_IDX] != MINOR_KEY

    def _divide_key_chord_type(_ks, major):
        if major:
            return _ks, ChordType.MAJOR.value
        return _ks[:CHORD_TYPE_IDX], ChordType.MINOR.value

    key_signature = getattr(meta_message, "key")
    _key, _chord_type = _divide_key_chord_type(key_signature, _is_major(key_signature))
    return _key.lower() + _chord_type


@with_default()
def get_time_signature_v2(meta_message: Optional[mido.MetaMessage]) -> str:
    attrs = ("numerator", "denominator")
    time_signature = "/".join(str(getattr(meta_message, attr)) for attr in attrs)
    return time_signature


def get_pitch_range_v2(midi_obj: mido.MidiFile, keyswitch_velocity: Optional[int] = None) -> str:
    def _get_avg_note(_tracks: List[pretty_midi.Instrument]):
        total = 0
        count = 0
        for track in _tracks:
            # pretty_midi ?????? ?????? ????????? Instrument ??? ???????????? ??????
            if track.name == CHORD_TRACK_NAME:
                continue
            for event in track.notes:
                if event.pitch == 0:
                    continue

                if keyswitch_velocity is not None:
                    if event.velocity != keyswitch_velocity:
                        total += event.pitch
                        count += 1
                else:
                    total += event.pitch
                    count += 1
        if not count:
            return None
        return total / count

    def _get_pitch_range(avg_pitch_range):
        indexer = {i: k for i, k in enumerate(PITCH_RANGE_CUT.keys())}
        bins = list(PITCH_RANGE_CUT.values())
        digitizer = functools.partial(np.digitize, bins=bins)
        return indexer[digitizer(avg_pitch_range)]

    with tempfile.NamedTemporaryFile(suffix=".mid") as f:
        midi_obj.save(filename=f.name)
        pt_midi = pretty_midi.PrettyMIDI(midi_file=f.name)

    avg_note = _get_avg_note(pt_midi.instruments)
    if avg_note is None:
        return UNKNOWN
    return _get_pitch_range(avg_note)


def get_inst_from_midi_v2(midi_path: Union[str, Path]) -> str:
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    if not midi_data.instruments:
        return UNKNOWN
    return str(midi_data.instruments[0].program)


def get_genre(midi_path: Union[str, Path], path_to_genre: Optional[Dict[str, str]] = None) -> str:
    if path_to_genre is not None:
        # TODO: ?????? ??????????????? ??????. ??????????????? ???
        pattern = re.compile("|".join(path_to_genre.keys()))
        # ????????????2 ???????????? `stem`??? ??????, ????????? ?????? ???????????? ?????? ?????? ??????
        matched = pattern.match(Path(midi_path).stem)
        if matched is None:
            raise UnprocessableMidiError("Could not find filename from meta csv")
        return path_to_genre.get(matched.group(), UNKNOWN)

    for genre in constants.GENRE_MAP:
        if genre in str(midi_path):
            return genre
    return constants.DEFAULT_GENRE


def get_velocity_range(
    midi_path: Union[str, Path], keyswitch_velocity: Optional[int] = None
) -> Tuple[Union[int, str], Union[int, str]]:
    pt_midi = pretty_midi.PrettyMIDI(str(midi_path))
    raw_track_to_channel = {
        track.name: get_channel(track) for track in mido.MidiFile(midi_path).tracks
    }
    track_to_channel = {
        track_name: channel
        for track_name, channel in raw_track_to_channel.items()
        if channel is not None
    }

    velocities = []
    for track in pt_midi.instruments:
        channel = track_to_channel[track.name]
        if channel == constants.CHORD_CHANNEL:
            continue
        for note in track.notes:
            if keyswitch_velocity is not None:
                if note.velocity != keyswitch_velocity:
                    velocities.append(note.velocity)
            else:
                velocities.append(note.velocity)

    if not velocities or max(velocities) == 0:
        return UNKNOWN, UNKNOWN
    return min(velocities), max(velocities)


def get_track_category_from_channel(track: mido.MidiTrack) -> str:
    """????????? ?????? ??????????????? ???????????????."""
    return constants.CHANNEL_FOR_MELODY.get(get_channel(track), UNKNOWN)


def get_meta_message_v2(meta_track: mido.MidiTrack, event_type: str) -> Optional[mido.MetaMessage]:
    messages = [event for event in copy.deepcopy(meta_track) if event.type == event_type]
    if not messages:
        return None
    return messages.pop()


def get_meta_message(meta_track: mido.MidiTrack, event_type: str) -> Union[mido.MetaMessage, str]:
    """?????? ????????? mido.MetaMessage??? ????????? return ?????????."""

    messages = [event for event in meta_track if event.type == event_type]

    if not messages:
        return NO_META_MESSAGE

    return messages.pop()


def encode_meta_info(midi_info: MidiInfo) -> List:
    meta = []
    if midi_info.bpm is not UNKNOWN:
        meta.append(midi_info.bpm + BPM_START_POINT)
    else:
        meta.append(BPM_UNKHOWN)

    if midi_info.audio_key is not UNKNOWN:
        meta.append(midi_info.audio_key + KEY_START_POINT)
    else:
        meta.append(KEY_UNKHOWN)

    if midi_info.time_signature is not UNKNOWN:
        meta.append(midi_info.time_signature + TS_START_POINT)
    else:
        meta.append(TS_UNKHOWN)

    if midi_info.pitch_range is not UNKNOWN:
        meta.append(midi_info.pitch_range + PITCH_RANGE_START_POINT)
    else:
        meta.append(PITCH_RANGE_UNKHOWN)

    if midi_info.num_measure == 4:
        meta.append(MEASURES_4)
    elif midi_info.num_measure == 8:
        meta.append(MEASURES_8)
    else:
        return None

    if midi_info.inst is not UNKNOWN:
        meta.append(midi_info.inst + INST_START_POINT)
    else:
        meta.append(INST_UNKHOWN)

    return meta


def split_train_val(input: np.array, target: np.ndarray, val_ratio: float) -> np.array:
    x_train, x_val, y_train, y_val = train_test_split(
        input, target, test_size=val_ratio, shuffle=True, random_state=2021
    )
    splits = {
        "input_train": x_train,
        "input_val": x_val,
        "target_train": y_train,
        "target_val": y_val,
    }
    return splits


def load_poza_meta(request_url: str, update_date:str, per_page: int = 1000) -> List[Dict[str, Any]]:
    return fetch_samples_from_backoffice(
        request_url=request_url, update_date=update_date, per_page=per_page, params={"auto_changed": False}
    )


def fetch_samples_from_backoffice(
    request_url: str, update_date: str, per_page: int = 100, params: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    page = 1
    result = []
    finished = False
    while not finished:
        has_next, samples = _fetch_samples(request_url, update_date, page=page, per_page=per_page, params=params)
        result.extend(samples)
        finished = not has_next
        page += 1
    return result


def _fetch_samples(
    url, date, page: int = 1, per_page: int = 100, params: Optional[Dict[str, Any]] = None
) -> Tuple[bool, List[Dict[str, Any]]]:
    request_params = {"page": page, "per_page": per_page, "created_at": date, "updated_at": date}
    if params is not None:
        request_params.update(params)

    res = requests.get(url, params=request_params)

    if res.status_code != http.HTTPStatus.OK:
        raise ValueError("Failed to fetch samples from backoffice")

    data = res.json()
    return data["has_next"], data["samples"]["samples"]


def concat_npy(encode_tmp_dir):
    npy_list = os.listdir(encode_tmp_dir)

    input_npy_list = [
        os.path.join(encode_tmp_dir, npy_file)
        for npy_file in npy_list
        if npy_file.startswith("input")
    ]
    target_npy_list = [
        os.path.join(encode_tmp_dir, npy_file)
        for npy_file in npy_list
        if npy_file.startswith("target")
    ]

    input_lst = []
    target_lst = []
    for input_npy_pth in input_npy_list:
        _input_npy = np.load(input_npy_pth, allow_pickle=True)
        input_lst.append(_input_npy)

    for target_npy_pth in target_npy_list:
        _target_npy = np.load(target_npy_pth, allow_pickle=True)
        target_lst.append(_target_npy)

    input_npy = np.array(input_lst)
    target_npy = np.array(target_lst)

    return input_npy, target_npy


def get_channel(track: mido.MidiTrack) -> int:
    for event in track:
        # `channel_prefix` ???????????? ????????? 15??? ???????????? ?????? ????????? ?????? ?????? ???????????? ??????
        if hasattr(event, "channel") and event.type != "channel_prefix":
            return event.channel


def apply_channel(midi_path: Union[str, Path], track_to_channel: Dict[str, int]) -> None:
    midi_obj = mido.MidiFile(midi_path)
    new_midi_obj = copy.deepcopy(midi_obj)
    new_tracks = []
    for track in midi_obj.tracks:
        new_track = mido.MidiTrack()
        for event in track:
            target_channel = track_to_channel.get(track.name)
            if hasattr(event, "channel") and target_channel is not None:
                event = event.copy(channel=target_channel)

            new_track.append(event)
        new_tracks.append(new_track)
    new_midi_obj.tracks = new_tracks
    new_midi_obj.save(midi_path)


def augment_by_key(midi_path: str, augmented_tmp_dir: str, key_change: int, data: str) -> Path:

    midi = pretty_midi.PrettyMIDI(midi_path)
    if data == "pozalabs" and len(midi.instruments) == 1:  # drum track
        return None
    midi_id = Path(midi_path).stem

    for idx, instrument in enumerate(midi.instruments):
        if instrument.name != constants.CHORD_TRACK_NAME:
            pitch_track_idx = idx
        else:
            chord_track_idx = idx
    pitch_track_notes = midi.instruments[pitch_track_idx].notes
    origin_key = int(midi.key_signature_changes[0].key_number)
    track_offset = midi.instruments[chord_track_idx].notes[0].start

    if origin_key < MINOR_KEY_OFFSET:
        try:
            midi.key_signature_changes[0].key_number = PRETTY_MAJOR_KEY[origin_key + key_change]
        except IndexError:
            midi.key_signature_changes[0].key_number = PRETTY_MAJOR_KEY[
                origin_key + key_change - len(PRETTY_MAJOR_KEY)
            ]
    else:
        origin_key = origin_key - MINOR_KEY_OFFSET
        try:
            midi.key_signature_changes[0].key_number = PRETTY_MINOR_KEY[origin_key + key_change]
        except IndexError:
            midi.key_signature_changes[0].key_number = PRETTY_MINOR_KEY[
                origin_key + key_change - len(PRETTY_MINOR_KEY)
            ]

    new_key_number = midi.key_signature_changes[0].key_number
    new_key = pretty_midi.key_number_to_key_name(new_key_number).lower().replace(" ", "")

    for note in pitch_track_notes:
        note.pitch = note.pitch + key_change
        note.start = note.start - track_offset
        note.end = note.end - track_offset

    if data == "pozalabs":
        midi.instruments.pop(chord_track_idx)
    try:
        midi.write(os.path.join(augmented_tmp_dir, midi_id + f"_{new_key}.mid"))
    except (AttributeError, ValueError):
        return None
    return os.path.join(augmented_tmp_dir, midi_id + f"_{new_key}.mid")


def get_avg_bpm(event_times: np.ndarray, tempo_infos: np.ndarray, end_time: float) -> int:
    def _normalize(_avg_bpm):
        return _avg_bpm - _avg_bpm % constants.BPM_INTERVAL

    if len(tempo_infos) == 1:
        return _normalize(tempo_infos[-1])

    event_times_with_end_time = np.concatenate([event_times, [end_time]])
    # `end_time`????????? ??? BPM ?????? ??????
    bpm_durations = np.diff(event_times_with_end_time)
    total_bpm = 0
    for duration, bpm in zip(bpm_durations, tempo_infos):
        total_bpm += duration * bpm

    avg_bpm = int(total_bpm / end_time)
    return _normalize(avg_bpm)


def augment_by_bpm(augment_tmp_midi_pth, augmented_dir, bpm_change) -> None:
    midi = pretty_midi.PrettyMIDI(augment_tmp_midi_pth)
    event_times, origin_bpm = midi.get_tempo_changes()

    if len(origin_bpm) > 1:
        origin_bpm = get_avg_bpm(event_times, origin_bpm, midi.get_end_time())

    mido_object = mido.MidiFile(augment_tmp_midi_pth)
    augment_midi_name = Path(augment_tmp_midi_pth).parts[-1].split(".")[0]
    for track in mido_object.tracks:
        for message in track:
            if message.type == "set_tempo":
                new_bpm = float(origin_bpm) + bpm_change * BPM_INTERVAL
                message.tempo = mido.bpm2tempo(new_bpm)
    try:
        mido_object.save(os.path.join(augmented_dir, augment_midi_name + f"_{int(new_bpm)}.mid"))
    except (AttributeError, ValueError):
        pass


def augment_data_map(
    midi_list: List,
    augmented_dir: str,
    augmented_tmp_dir: str,
    data: str,
) -> None:
    for midi_path in midi_list:
        for key_change in range(-NUM_KEY_AUGMENT, NUM_KEY_AUGMENT):
            augment_tmp_midi_pth = augment_by_key(midi_path, augmented_tmp_dir, key_change, data)
            if augment_tmp_midi_pth is not None:
                for bpm_change in range(-NUM_BPM_AUGMENT, NUM_BPM_AUGMENT+1): #?????? ????????? ?????? +1
                    augment_by_bpm(augment_tmp_midi_pth, augmented_dir, bpm_change)


def augment_data(
    midi_path: Union[str, Path],
    augmented_dir: Union[str, Path],
    augmented_tmp_dir: Union[str, Path],
    data: str,
    num_cores: int,
) -> None:

    midifiles = []

    for _, (dirpath, _, filenames) in enumerate(os.walk(midi_path)):
        midi_extensions = [".mid", ".MID", ".MIDI", ".midi"]
        for ext in midi_extensions:
            tem = [os.path.join(dirpath, _) for _ in filenames if _.endswith(ext)]
            if tem:
                midifiles += tem

    split_midi = np.array_split(np.array(midifiles), num_cores)
    split_midi = [x.tolist() for x in split_midi]
    parmap.map(
        augment_data_map,
        split_midi,
        augmented_dir,
        augmented_tmp_dir,
        data,
        pm_pbar=True,
        pm_processes=num_cores,
    )


def find_chord_name(notes: List[music21.note.Note]) -> str:
    midi_notes = [note.pitch.midi for note in notes]
    notes_in_degree = get_notes_in_degree(midi_notes)
    chord_name = find_chord_name_for_notes_in_degree(notes_in_degree)

    if chord_name is None:
        raise InvalidMidiError(InvalidMidiErrorMessage.invalid_chord.format(midi_notes))

    root_pitch_class = notes[0].pitch.pitchClass
    root_name = constants.VAL_NOTE_DICT[root_pitch_class][0]
    return root_name + chord_name


def get_notes_in_degree(midi_notes: List[int]) -> List[int]:
    """?????? ?????? (0~127)?????? ????????? ?????? ?????? ????????? ?????? ????????? ???????????? ??????"""
    result = []
    for note in midi_notes:
        note_in_degree = (note - midi_notes[0]) % constants.DEGREE
        if note_in_degree not in result:
            result.append(note_in_degree)
    return result


def find_chord_name_for_notes_in_degree(notes_in_degree: List[int]) -> Optional[str]:
    for name, chord_notes in constants.CHORD_NOTE.items():
        if chord_notes == notes_in_degree:
            return name


def divide_chord_into_unit_beat(
    chord: str, quarter_length: float, unit_beat: float = constants.EIGHTH_NOTE_BEATS
) -> List[str]:
    if not check_divisible_by_unit_beat(quarter_length, unit_beat):
        raise InvalidMidiError(InvalidMidiErrorMessage.invalid_chord_note_length.value)

    num_units = int(quarter_length / unit_beat)
    return [chord] * num_units


def check_divisible_by_unit_beat(
    quarter_length: Union[Fraction, float], unit_beat: float = constants.EIGHTH_NOTE_BEATS
) -> bool:
    """????????? quarter length ??? ?????? ?????? (??????: 8??? ??????)??? ???????????? ???????????? ?????? (quarter_length:
    quarter length ?????? ??????: https://web.mit.edu/music21/doc/moduleReference/moduleBase.html#music21.base.Music21Object.quarterLength  # noqa: E501
    """
    return float(quarter_length / unit_beat).is_integer()


def find_chord_name_for_midi(added_notes: List[int]) -> str:
    """?????? ????????? ?????? ?????? ????????? ?????? ??????"""

    def _change_chord_name_form(_chord_name: str) -> str:
        return "".join(constants.CHORD_NAME_FORM.get(char, char) for char in _chord_name)

    notes_in_degree = []
    for note in added_notes:
        note_in_degree = (note - added_notes[0]) % constants.DEGREE
        if note_in_degree not in notes_in_degree:
            notes_in_degree.append(note_in_degree)

    result = None
    for name, chord_notes in constants.CHORD_NOTE.items():
        if chord_notes == notes_in_degree:
            result = name

    if result is None:
        raise InvalidMidiError(InvalidMidiErrorMessage.invalid_chord.format(notes_in_degree))

    result = constants.VAL_NOTE_DICT.get(added_notes[0] % 12)[0] + result
    return _change_chord_name_form(result)


def get_num_measures_from_midi_v2(
    midi_path: Union[str, Path], track_name: Optional[str] = None
) -> int:
    """???????????? ?????? ?????? ???????????? ??????"""

    def _get_track(tracks):
        if track_name is None:
            return tracks[0]
        for t in tracks:
            if t.name == track_name:
                return t

    pt_midi = pretty_midi.PrettyMIDI(str(midi_path))
    time_signature: pretty_midi.TimeSignature = pt_midi.time_signature_changes[-1]
    coordination = time_signature.numerator / time_signature.denominator
    ticks_per_measure = pt_midi.resolution * DEFAULT_NUM_BEATS * coordination

    track = _get_track(pt_midi.instruments)
    if track is None:
        raise InvalidMidiError(InvalidMidiErrorMessage.chord_track_not_found.value)

    notes = track.notes
    duration_tick = pt_midi.time_to_tick(notes[-1].end - notes[0].start)
    return math.ceil(duration_tick / ticks_per_measure)


def get_chord_streams(file_path: Union[str, Path]) -> List[music21.stream.Stream]:
    """?????? ???????????? ?????? ????????? ????????? ??? music21.stream.Stream ???????????? ???????????? ??????.
    ?????? ?????? ?????? ????????? ???????????? ?????? ????????? ????????? ???????????? ???????????? ??? ??????
    """

    def _get_chord_stream(_music21_midi_track: music21.midi.MidiTrack) -> music21.stream.Stream:
        return music21.midi.translate.midiTrackToStream(
            mt=_music21_midi_track, ticksPerQuarter=music21_midi.ticksPerQuarterNote
        )

    def _remove_rest_before_region_start(_stream: music21.stream.Stream) -> music21.stream.Stream:
        start_idx = 0
        for idx, event in enumerate(_stream):
            if not isinstance(event, music21.note.Rest):
                start_idx = idx
                break
        return _stream[start_idx:]

    music21_midi = read_midi_music21(file_path)
    return [
        _remove_rest_before_region_start(_get_chord_stream(track))
        for track in get_chord_tracks(music21_midi)
    ]


def read_midi_music21(file_path: Union[str, Path]) -> music21.midi.MidiFile:
    mf = music21.midi.MidiFile()
    mf.open(file_path)
    mf.read()
    mf.close()
    return mf


def get_chord_tracks(music21_midi: music21.midi.MidiFile) -> List[music21.midi.MidiTrack]:
    if music21_midi.file is None:
        raise ValueError("Midi file is not read. Read midi file first")

    result = []
    for track in music21_midi.tracks:
        for event in track.events:
            if (
                event.type == music21.midi.MetaEvents.SEQUENCE_TRACK_NAME
                and event.data.decode("utf-8") == CHORD_TRACK_NAME
            ):
                result.append(track)

    if not result:
        raise InvalidMidiError(InvalidMidiErrorMessage.chord_track_not_found.value)
    return result


def get_cc_values(midi_track: mido.MidiTrack, control_change: str) -> Optional[List[int]]:
    """?????? ????????? control change ?????? ???????????? ??????.
    control_change ??? constants.CONTROL_CHANGE_DICT ??? ??? ?????? ????????? ????????? ?????????.
    """
    if control_change not in constants.CONTROL_CHANGE_DICT.keys():
        raise ValueError

    cc_parameter = constants.CONTROL_CHANGE_DICT[control_change]
    available_cc = set(
        [message.control for message in midi_track if message.type == "control_change"]
    )
    if cc_parameter not in available_cc:
        return None

    cc_values = [
        message.value
        for message in midi_track
        if message.type == "control_change" and message.control == cc_parameter
    ]
    return cc_values


def get_modulation_range(midi_path: Union[str, Path]) -> Tuple[Union[int, str], Union[int, str]]:
    mido_obj = mido.MidiFile(midi_path)
    for midi_track in mido_obj.tracks[1:]:
        is_chord_track = [message.name for message in midi_track if message.type == "track_name"]
        if constants.CHORD_TRACK_NAME in is_chord_track:
            continue

        modulations = get_cc_values(midi_track, control_change="modulation")
        if not modulations:
            return constants.UNKNOWN, constants.UNKNOWN
        return min(modulations), max(modulations)


def get_expression_range(midi_path: Union[str, Path]) -> Tuple[Union[int, str], Union[int, str]]:
    mido_obj = mido.MidiFile(midi_path)
    for midi_track in mido_obj.tracks[1:]:
        is_chord_track = [message.name for message in midi_track if message.type == "track_name"]
        if constants.CHORD_TRACK_NAME in is_chord_track:
            continue

        expressions = get_cc_values(midi_track, control_change="expression")
        if not expressions:
            return constants.UNKNOWN, constants.UNKNOWN
        return min(expressions), max(expressions)


def get_sustain_range(midi_path: Union[str, Path]) -> Tuple[Union[int, str], Union[int, str]]:
    mido_obj = mido.MidiFile(midi_path)
    for midi_track in mido_obj.tracks[1:]:
        is_chord_track = [message.name for message in midi_track if message.type == "track_name"]
        if constants.CHORD_TRACK_NAME in is_chord_track:
            continue

        sustains = get_cc_values(midi_track, control_change="sustain")
        if not sustains:
            return constants.UNKNOWN, constants.UNKNOWN
        return min(sustains), max(sustains)


def combine_dataset(
    reddit_pth: str,
    poza_pth: str,
    poza2_pth: str,
    reddit_num_sampling: int,
    num_meta: int,
    val_ratio: float,
    save_dir: str,
) -> None:
    def _concat_npy(pth: str) -> Union[np.ndarray, np.ndarray]:
        def _gather(_prefix):
            return sorted(
                str(i) for i in Path(pth).rglob("**/*/*.npy") if i.stem.startswith(_prefix)
            )

        def _concat(npy_list):
            res = None
            for f in npy_list:
                tmp = np.load(f, allow_pickle=True)
                if res is not None:
                    res = np.concatenate((res, tmp))
                else:
                    res = tmp
            return res

        return _concat(_gather("input")), _concat(_gather("target"))

    def _concat_meta_note(input: np.ndarray, target: np.ndarray) -> np.ndarray:
        all = []
        for meta, note in zip(input, target):
            all.append(np.concatenate((meta, note)))
        return np.array(all)

    def _combine_data(
        reddit: np.ndarray,
        poza: np.ndarray,
        poza2: np.ndarray,
        num_meta: int,
        reddit_num_sampling: int,
    ) -> Union[np.ndarray, np.ndarray]:
        reddit_sample = np.random.choice(reddit, reddit_num_sampling, replace=False)
        total = np.concatenate((reddit_sample, poza, poza2))
        meta = []
        note = []
        for i in total:
            meta.append(i[: num_meta + 1])
            note.append(i[num_meta + 1 :])
        return np.array(meta), np.array(note)

    # load datasets
    input_reddit = np.load(f"{reddit_pth}/input_train.npy", allow_pickle=True)
    target_reddit = np.load(f"{reddit_pth}/target_train.npy", allow_pickle=True)
    input_poza2, target_poza2 = _concat_npy(poza2_pth)
    input_poza, target_poza = _concat_npy(poza_pth)

    # concat meta & note respectively
    reddit_all = _concat_meta_note(input_reddit, target_reddit)
    poza_all = _concat_meta_note(input_poza, target_poza)
    poza2_all = _concat_meta_note(input_poza2, target_poza2)

    # combine, mix, split 3 datasets
    meta, note = _combine_data(reddit_all, poza_all, poza2_all, num_meta, reddit_num_sampling)
    splits = split_train_val(meta, note, val_ratio=val_ratio)

    # save
    for split_name, data_split in splits.items():
        np.save(str(Path(save_dir).joinpath(split_name)), data_split)
