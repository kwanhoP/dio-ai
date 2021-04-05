import copy
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union, cast

import mido
import music21
import numpy as np
import parmap
import pretty_midi

from dioai.exceptions import UnprocessableMidiError
from dioai.preprocessor import utils
from dioai.preprocessor.utils import constants

# Tempo, Key, Time Signature가 너무 자주 바뀌는 경우는 학습에 이용하지 않음
MAXIMUM_CHANGE = 8
UNIT_BPM = 60


@dataclass
class TrackIgnoreStrategy:
    """미디 청크 시 무시할 트랙을 선택하기 위한 객체

    Attributes:
        source: `str`. 무시할 트래을 판단할 기준 (가능한 값: track, channel, program, ...)
        values: `Iterable[Union[str, int]]`. `source`를 기준으로, 무시할 트랙들
    """

    source: str
    values: Any


TRACK_IGNORE_STRATEGIES = {
    "reddit": TrackIgnoreStrategy(source="program", values=constants.INSTRUMENT_NOT_FOR_MELODY),
    "pozalabs2": TrackIgnoreStrategy(source="channel", values=constants.CHANNEL_NOT_FOR_MELODY),
}


def filter_melody_tracks_reddit(midi_path: Union[str, Path]) -> str:
    def _get_program(_track: mido.MidiTrack):
        for _event in _track:
            if hasattr(_event, "program_change"):
                return _event.program
        return 0

    programs_not_melody = TRACK_IGNORE_STRATEGIES["reddit"].values
    midi_obj = mido.MidiFile(midi_path)
    new_tracks = []
    for track in midi_obj.tracks:
        if not track:
            continue

        channel = utils.get_channel(track)
        program = _get_program(track)
        if channel == constants.DRUM_CHANNEL or program in programs_not_melody:
            continue

        new_tracks.append(track)
    midi_obj.tracks = new_tracks
    midi_obj.save(midi_path)
    return str(midi_path)


def filter_melody_tracks_pozalabs2(midi_path: Union[str, Path]) -> str:
    midi_path = Path(midi_path)
    midi_obj = mido.MidiFile(midi_path)
    new_midi_obj = copy.deepcopy(midi_obj)
    midi_tracks = new_midi_obj.tracks
    channels_not_melody = TRACK_IGNORE_STRATEGIES["pozalabs2"].values
    new_tracks = [midi_tracks[0]]
    for track in midi_tracks[1:]:
        track_name = track.name.strip()
        channel = utils.get_channel(track)
        if track_name in channels_not_melody.keys() or channel in channels_not_melody.values():
            continue
        new_tracks.append(track)

    new_midi_path = midi_path
    new_midi_obj.tracks = new_tracks
    new_midi_obj.save(new_midi_path)
    return str(new_midi_path)


TrackFilterFuncType = Callable[[Union[str, Path]], str]
TRACK_FILTER_FUNCS: Dict[str, TrackFilterFuncType] = {
    "reddit": filter_melody_tracks_reddit,
    "pozalabs2": filter_melody_tracks_pozalabs2,
}


def get_avg_bpm(event_times: np.ndarray, tempo_infos: np.ndarray, end_time: float) -> int:
    def _normalize(_avg_bpm):
        return _avg_bpm - _avg_bpm % constants.BPM_INTERVAL

    if len(tempo_infos) == 1:
        return _normalize(int(tempo_infos[-1]))

    event_times_with_end_time = np.concatenate([event_times, [end_time]])
    # `end_time`까지의 각 BPM 지속 시간
    bpm_durations = np.diff(event_times_with_end_time)
    total_bpm = 0
    for duration, bpm in zip(bpm_durations, tempo_infos):
        total_bpm += duration * bpm

    avg_bpm = int(total_bpm / end_time)
    return _normalize(avg_bpm)


def normalize_bpm(
    midi_path: Union[str, Path], output_dir: Union[str, Path], max_bpm_changes: int = MAXIMUM_CHANGE
) -> Tuple[str, int]:
    """미디의 BPM을 평균 BPM으로 정규화합니다."""
    try:
        pt_midi = pretty_midi.PrettyMIDI(str(midi_path))
    except (
        OSError,
        KeyError,
        ValueError,
        EOFError,
        IndexError,
        ZeroDivisionError,
    ) as exc:
        raise UnprocessableMidiError("Failed to read midi using `pretty_midi`") from exc

    if not pt_midi.get_end_time():
        raise UnprocessableMidiError("Empty midi")

    event_times, tempo_infos = pt_midi.get_tempo_changes()
    if len(event_times) > max_bpm_changes:
        raise UnprocessableMidiError(
            f"Too many bpm changes: {len(event_times)} > {max_bpm_changes}"
        )

    avg_bpm = get_avg_bpm(event_times, tempo_infos, end_time=pt_midi.get_end_time())
    midi_obj = mido.MidiFile(midi_path)
    for track in midi_obj.tracks:
        for message in track:
            if message.type == "set_tempo":
                message.tempo = mido.bpm2tempo(avg_bpm)

    midi_path_with_avg_bpm = str(Path(output_dir).joinpath(f"{Path(midi_path).stem}_{avg_bpm}.mid"))
    try:
        midi_obj.save(midi_path_with_avg_bpm)
    except (AttributeError, ValueError):
        raise UnprocessableMidiError(f"Failed to save midi to {midi_path_with_avg_bpm}")
    return midi_path_with_avg_bpm, avg_bpm


def chunk_track(
    instrument: pretty_midi.Instrument,
    steps_per_sec: float,
    longest_allowed_space: int,
    minimum_chunk_length: int,
    step_in_sec: float,
    truncate_under_nth_decimal: int,
) -> List[Tuple[List[pretty_midi.Note], float]]:
    """
    각 chunk간의 time space 를 측정하려면, 여러 방법이 있겠지만,
    나는 전체 곡의 길이를 0.1초 단위로 나눈다음 (각 0.1초 구간을 bin이라고 한다)
    note들을 순회하면서 각 note가 걸치는 bin들을 True로 바꾼다(채운다).
    모든 note들에 대해 bin을 채우는 일을 다 하면,
    note 사이의 시간 간격이 LONGEST_ALLOWED_SPACE 이상이면 별개의 chunk로 취급하며,
    해당 chunk의 길이가 MINIMUM_CHUNK_LENGTH 이상일 때만 최종적으로 chunk로 인정하여,
    이렇게 최종적으로 얻어진 각 chunk에 해당하는 note들을 추려 하나의 midi파일로 저장한다.
    저장되는 midi파일의 이름은 "원본미디파일이름_{instrument_index}_{chunk_index_in_the_instrument} 이다"
    a bin in time_bins will be filled(turn to True) if any note occupies the bin
    time_bins is quantized into 1/STEPS_PER_SEC sec
    """

    # unit is 1/STEPS_PER_SEC
    time_bins_sec = np.arange(0, instrument.get_end_time() + step_in_sec, step_in_sec).tolist()
    time_bins_sec = [
        truncate(sec, truncate_under_nth_decimal) for sec in cast(List[float], time_bins_sec)
    ]
    time_bins = [False] * len(time_bins_sec)
    # time_bins = {str(floor_to_1st_place_decimal): False for time_bin in time_bins}
    for note in instrument.notes:
        note_start = truncate(note.start, truncate_under_nth_decimal)
        note_end = truncate(note.end, truncate_under_nth_decimal)
        note_start_sec_index = time_bins_sec.index(note_start)
        note_end_sec_index = time_bins_sec.index(note_end)
        for sec_index in range(note_start_sec_index, note_end_sec_index + 1):
            time_bins[sec_index] = True

    new_notes_per_instrument = []
    chunk_start_idx = -1
    chunk_end_idx = -1
    note_idx_to_start_checking = 0
    counter = 0
    for time_idx, time_bin in enumerate(time_bins):
        if time_bin:
            if (
                counter >= steps_per_sec * longest_allowed_space or chunk_start_idx == -1
            ):  # when new chunk bigins or when the first chunk from instrument begins
                chunk_start_idx = time_idx
            chunk_end_idx = time_idx
            counter = 0
        else:
            counter += 1

        if counter == steps_per_sec * longest_allowed_space or time_idx == len(time_bins) - 1:
            if chunk_end_idx - chunk_start_idx >= steps_per_sec * minimum_chunk_length:

                chunk_start_sec = float(time_bins_sec[chunk_start_idx])
                chunk_end_sec = float(time_bins_sec[chunk_end_idx])

                note_chunk_start_idx = -1
                start_note_sec = None
                for note_idx in range(note_idx_to_start_checking, len(instrument.notes)):
                    if (
                        note_chunk_start_idx == -1
                        and instrument.notes[note_idx].start >= chunk_start_sec
                    ):
                        note_chunk_start_idx = note_idx
                        start_note_sec = instrument.notes[note_chunk_start_idx].start
                    if instrument.notes[note_idx].start > chunk_end_sec + step_in_sec or (
                        note_idx == len(instrument.notes) - 1
                    ):  # or 뒤의 조건을 빠뜨리면 instrument의 끝과 맞닿아 있는 chunk가 포함이 안됨.
                        # pull timelilnes of notes to start of first note
                        for track_note in instrument.notes[note_chunk_start_idx:note_idx]:
                            track_note.start -= start_note_sec
                            track_note.end -= start_note_sec
                        new_notes_per_instrument.append(
                            (instrument.notes[note_chunk_start_idx:note_idx], start_note_sec)
                        )
                        note_idx_to_start_checking = note_idx
                        break
    return new_notes_per_instrument


def truncate(f: float, n: int, dtype="str") -> Union[str, float]:
    """Truncates/pads a float f to n decimal places without rounding"""
    s = "{}".format(f)
    if "e" in s or "E" in s:
        truncated_str = "{0:.{1}f}".format(f, n)
        if dtype == "float":
            return float(truncated_str)
        else:
            return truncated_str
    i, _, d = s.partition(".")
    truncated_str = ".".join([i, (d + "0" * n)[:n]])
    if dtype == "float":
        return float(truncated_str)
    else:
        return truncated_str


def get_chord_track(
    midi_obj: mido.MidiFile, pt_midi: pretty_midi.PrettyMIDI
) -> pretty_midi.Instrument:
    for idx, track in enumerate(midi_obj.tracks[1:]):
        if utils.get_channel(track) == constants.CHORD_CHANNEL:
            chord_track_name = track.name
            instrument = pt_midi.instruments[idx]
            if chord_track_name != instrument.name:
                raise RuntimeError(
                    f"Track name of `mido.MidiFile` and `pretty_midi.PrettyMidi` must be same,"
                    f"but got: {chord_track_name} != {instrument.name}"
                )
            return instrument

    raise UnprocessableMidiError(f"Could not find chord track from {midi_obj.filename}")


def chunk_chord_track(
    chord_track: pretty_midi.Instrument,
    chunk_start: float,
    chunk_end: float,
    bpm: int,
    time_signature_changes: List[pretty_midi.TimeSignature],
    time_to_tick_func: Callable[[float], int],
) -> pretty_midi.Instrument:
    def _get_unique_ts_changes(_time_signature_changes):
        _unique_time_signatures = []
        times_in_ts = set()
        for _ts_change in _time_signature_changes:
            time_in_ts = _ts_change.time

            if time_in_ts in times_in_ts:
                _unique_time_signatures.pop()
            else:
                times_in_ts.add(time_in_ts)

            _unique_time_signatures.append(_ts_change)
        return _unique_time_signatures

    time_signature_changes = _get_unique_ts_changes(time_signature_changes)

    def _get_seconds_per_measure(_ts: pretty_midi.TimeSignature) -> float:
        music21_ts = music21.meter.TimeSignature(f"{_ts.numerator}/{_ts.denominator}")
        bar_quarter_length = music21_ts.barDuration.quarterLength
        return UNIT_BPM / bpm * bar_quarter_length

    def _get_seconds_per_measure_changes(_time_signature_changes):
        _seconds_per_measure_changes = []
        for _ts in _time_signature_changes:
            _seconds_per_measure_changes.append((_ts.time, _get_seconds_per_measure(_ts)))
        return _seconds_per_measure_changes

    def _get_measure_start(_chunk_start):
        _seconds_per_measure_changes = _get_seconds_per_measure_changes(time_signature_changes)
        for _i, (_time, _seconds_per_measures) in enumerate(_seconds_per_measure_changes):
            if Decimal(_chunk_start) >= Decimal(_time):
                _seconds_until = list(zip(*_seconds_per_measure_changes[:_i]))
                _measure_end = sum(_seconds_until[0]) if _seconds_until else 0
                while time_to_tick_func(_chunk_start) > time_to_tick_func(_measure_end):
                    _measure_end += _seconds_per_measures
                return max(0.0, _measure_end - _seconds_per_measures)

    def _get_measure_end(_chunk_end):
        _seconds_per_measure_changes = _get_seconds_per_measure_changes(time_signature_changes)
        for _i, (_time, _seconds_per_measures) in enumerate(_seconds_per_measure_changes):
            if Decimal(_chunk_end) >= Decimal(_time):
                _seconds_until = list(zip(*_seconds_per_measure_changes[:_i]))
                _measure_end = sum(_seconds_until[0]) if _seconds_until else 0
                while time_to_tick_func(_chunk_end) > time_to_tick_func(_measure_end):
                    _measure_end += _seconds_per_measures
                return _measure_end

    def _check_note_in_chunk(_note, _chunk_start_measure_start, _chunk_end_measure_end):
        return time_to_tick_func(_note.start) >= time_to_tick_func(
            _chunk_start_measure_start
        ) and time_to_tick_func(_note.end) <= time_to_tick_func(_chunk_end_measure_end)

    def _get_first_matched_chord_note(_new_chord_track, _chunk_start, _chunk_end):
        for _note in _new_chord_track.notes:
            if _check_note_in_chunk(_note, _chunk_start, _chunk_end):
                return _note

    measure_start = _get_measure_start(chunk_start)
    new_chord_track = copy.deepcopy(chord_track)
    new_notes = []

    first_chord_note = _get_first_matched_chord_note(new_chord_track, chunk_start, chunk_end)
    if first_chord_note is None:
        raise UnprocessableMidiError("Chord notes do not exist for this chunk. Maybe invalid midi")

    is_incomplete_measure = time_to_tick_func(measure_start) < time_to_tick_func(
        first_chord_note.start
    )

    chord_track_start = _get_measure_end(chunk_start) if is_incomplete_measure else measure_start
    chord_track_end = _get_measure_end(chunk_end)
    for note in new_chord_track.notes:
        if _check_note_in_chunk(note, chord_track_start, chord_track_end):
            new_note = copy.deepcopy(note)
            new_note.start -= first_chord_note.start
            new_note.end -= first_chord_note.start
            new_notes.append(new_note)

    new_chord_track.notes = new_notes
    return new_chord_track


def chunk_midi_map(
    midi_paths,
    steps_per_sec,
    longest_allowed_space,
    minimum_chunk_length,
    chunked_midi_path,
    tmp_midi_dir,
    dataset_name: str,
    preserve_chord_track: bool = False,
) -> None:
    # 소수점 아래 몇 자리 이후를 버릴지
    truncate_under_fp = len(str(steps_per_sec)) - 1
    for filename in midi_paths:
        filename_wo_ext = Path(filename).stem
        try:
            _midi_path_by_avg_tempo, average_tempo = normalize_bpm(filename, tmp_midi_dir)
        except UnprocessableMidiError:
            continue

        track_filter_func = TRACK_FILTER_FUNCS[dataset_name]
        midi_path_with_valid_tracks = track_filter_func(_midi_path_by_avg_tempo)

        midi_obj_with_valid_tracks = mido.MidiFile(midi_path_with_valid_tracks)
        mido_tracks_wo_meta = midi_obj_with_valid_tracks.tracks[1:]

        midi_data = pretty_midi.PrettyMIDI(midi_path_with_valid_tracks)
        if not midi_data.instruments:
            continue

        for inst_idx, instrument in enumerate(midi_data.instruments):
            mido_track_name = mido_tracks_wo_meta[inst_idx].name
            if mido_track_name != instrument.name:
                raise RuntimeError(
                    f"Track name of `mido.MidiFile` and `pretty_midi.PrettyMidi` must be same,"
                    f"but got: {mido_track_name} != {instrument.name}"
                )

            if (
                utils.get_channel(mido_tracks_wo_meta[inst_idx]) == constants.CHORD_CHANNEL
                or instrument.name == constants.CHORD_TRACK_NAME
            ):
                continue

            new_notes_per_instrument = chunk_track(
                instrument=instrument,
                steps_per_sec=steps_per_sec,
                longest_allowed_space=longest_allowed_space,
                minimum_chunk_length=minimum_chunk_length,
                step_in_sec=truncate(1.0 / steps_per_sec, truncate_under_fp, dtype="float"),
                truncate_under_nth_decimal=truncate_under_fp,
            )

            chunked_chord_track = None
            for i, (notes, chunk_start) in enumerate(new_notes_per_instrument):
                mido_track = mido_tracks_wo_meta[inst_idx]
                track_to_channel = {mido_track.name: utils.get_channel(mido_track)}

                if not notes:
                    continue

                if preserve_chord_track:
                    try:
                        chord_track = get_chord_track(midi_obj_with_valid_tracks, midi_data)
                        track_to_channel[chord_track.name] = constants.CHORD_CHANNEL
                        chunked_chord_track = chunk_chord_track(
                            chord_track,
                            chunk_start=chunk_start,
                            chunk_end=notes[-1].end + chunk_start,
                            bpm=average_tempo,
                            time_signature_changes=midi_data.time_signature_changes,
                            time_to_tick_func=midi_data.time_to_tick,
                        )
                    except UnprocessableMidiError:
                        continue

                    if not chunked_chord_track.notes:
                        continue

                new_midi_object = pretty_midi.PrettyMIDI(
                    resolution=midi_data.resolution, initial_tempo=average_tempo
                )

                ks_list = midi_data.key_signature_changes
                ts_list = midi_data.time_signature_changes

                if len(ks_list) > MAXIMUM_CHANGE or len(ts_list) > MAXIMUM_CHANGE:
                    break

                # ks 가 변화하지 않는 경우 default값으로 설정 필요
                if ks_list:
                    new_midi_object.key_signature_changes = ks_list
                # ts 가 변화하지 않는 경우 default값으로 설정 필요
                if ts_list:
                    new_midi_object.time_signature_changes = ts_list

                new_instrument = pretty_midi.Instrument(instrument.program, name=instrument.name)
                new_instrument.notes = notes
                new_midi_object.instruments.append(new_instrument)
                if preserve_chord_track:
                    new_midi_object.instruments.append(chunked_chord_track)

                chunked_midi_filename = Path(chunked_midi_path).joinpath(
                    f"{filename_wo_ext}_{inst_idx}_{instrument.program}_{i}.mid"
                )
                new_midi_object.write(str(chunked_midi_filename))
                utils.apply_channel(chunked_midi_filename, track_to_channel)


def chunk_midi(
    steps_per_sec,
    longest_allowed_space,
    minimum_chunk_length,
    midi_dataset_path,
    chunked_midi_path,
    tmp_midi_dir,
    num_cores: int,
    dataset_name: str,
    preserve_chord_track: bool = False,
) -> None:
    midi_paths = [
        str(filename)
        for filename in Path(midi_dataset_path).rglob("**/*")
        if (
            Path(chunked_midi_path).name != filename.parent.name
            and Path(tmp_midi_dir).name != filename.parent.name
        )
        and filename.suffix in constants.MIDI_EXTENSIONS
    ]

    split_midi = np.array_split(np.array(sorted(midi_paths)), num_cores)
    split_midi = [x.tolist() for x in split_midi]
    parmap.map(
        chunk_midi_map,
        split_midi,
        steps_per_sec=steps_per_sec,
        longest_allowed_space=longest_allowed_space,
        minimum_chunk_length=minimum_chunk_length,
        chunked_midi_path=chunked_midi_path,
        tmp_midi_dir=tmp_midi_dir,
        dataset_name=dataset_name,
        preserve_chord_track=preserve_chord_track,
        pm_pbar=True,
        pm_processes=num_cores,
    )
