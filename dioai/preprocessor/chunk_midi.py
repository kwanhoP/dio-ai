import os

import mido
import numpy as np
import parmap
import pretty_midi
from tqdm import tqdm

from .constants import BPM_INTERVAL, INSTRUMENT_NOT_FOR_MELODY, NUM_CORES

# Tempo, Key, Time Signature가 너무 자주 바뀌는 경우는 학습에 이용하지 않음
MAXIMUM_CHANGE = 8


def chunk_midi_map(
    midifiles,
    steps_per_sec,
    longest_allowed_space,
    minimum_chunk_length,
    chunked_midi_path,
    tmp_midi_dir,
):
    def _truncate(f: float, n: int, dtype="str") -> float:
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

    TRUNCATE_UNDER_NTH_DECIMAL = len(str(steps_per_sec)) - 1  # 소수점 아래 몇 자리 이후를 버릴지
    STEP_IN_SEC = _truncate(1.0 / steps_per_sec, TRUNCATE_UNDER_NTH_DECIMAL, dtype="float")
    for filename in tqdm(midifiles):
        try:
            midi_data = pretty_midi.PrettyMIDI(filename)
        except (
            OSError,
            KeyError,
            ValueError,
            mido.midifiles.meta.KeySignatureError,
            EOFError,
            IndexError,
        ):
            continue

        # Get Average Tempo
        event_times, tempo_infos = midi_data.get_tempo_changes()

        if len(event_times) > MAXIMUM_CHANGE:
            continue

        total_tempo = 0
        for i, cur_tempo in enumerate(tempo_infos):
            if i == 0:
                prev_tempo = cur_tempo
                continue
            tempo_duration = event_times[i] - event_times[i - 1]
            total_tempo += tempo_duration * prev_tempo
            prev_tempo = cur_tempo

        if midi_data.get_end_time() == 0:
            continue

        tempo_duration = midi_data.get_end_time() - event_times[i]
        total_tempo += tempo_duration * prev_tempo

        if total_tempo == 0:
            average_tempo = cur_tempo
        else:
            average_tempo = int(total_tempo / midi_data.get_end_time())

        # 학습시킬때 BPM_INTERVAL 단위로 입력값이 들어가기 떄문에
        # tick의 오차를 해결하기 위해서 필요한 부분 이부분이 해결되어야 마디 길이에 정확히 설정 가능
        average_tempo = average_tempo - average_tempo % BPM_INTERVAL

        del midi_data

        filename_without_extension = os.path.splitext(filename.split("/")[-1])[0]

        # set default
        # pretty_midi에서 Tempo의 반영이 안되어, mido를 통한 Tempo 반영
        mido_object = mido.MidiFile(filename)
        for track in mido_object.tracks:
            for message in track:
                if message.type == "set_tempo":
                    message.tempo = mido.bpm2tempo(average_tempo)

        mido_object.save(
            os.path.join(tmp_midi_dir, filename_without_extension + f"_{average_tempo}.mid")
        )
        midi_data = pretty_midi.PrettyMIDI(
            os.path.join(tmp_midi_dir, filename_without_extension + f"_{average_tempo}.mid")
        )

        # First, remove instrument track not necessary for generating melody
        instrument_idx_to_remove = []
        for inst_idx, instrument in enumerate(midi_data.instruments):
            # if instrument is drum or not suitable for melody
            if instrument.is_drum or instrument.program in INSTRUMENT_NOT_FOR_MELODY:
                instrument_idx_to_remove.append(inst_idx)
            # if instrument has no notes in it, remove too.
            if not instrument.notes:
                instrument_idx_to_remove.append(inst_idx)

        midi_data.instruments = [
            instrument
            for inst_idx, instrument in enumerate(midi_data.instruments)
            if inst_idx not in instrument_idx_to_remove
        ]
        # if no instrument left after removing not necessary instruments, process next midi file
        if not midi_data.instruments:
            continue

        for inst_idx, instrument in enumerate(midi_data.instruments):
            new_notes_per_instrument = []
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
            time_bins_sec = np.arange(
                0, instrument.get_end_time() + STEP_IN_SEC, STEP_IN_SEC
            ).tolist()  # unit is 1/STEPS_PER_SEC
            time_bins_sec = [_truncate(sec, TRUNCATE_UNDER_NTH_DECIMAL) for sec in time_bins_sec]
            time_bins = [False] * len(time_bins_sec)
            # time_bins = {str(floor_to_1st_place_decimal): False for time_bin in time_bins}
            for note in instrument.notes:
                note_start = _truncate(note.start, TRUNCATE_UNDER_NTH_DECIMAL)
                note_end = _truncate(note.end, TRUNCATE_UNDER_NTH_DECIMAL)
                note_start_sec_index = time_bins_sec.index(note_start)
                note_end_sec_index = time_bins_sec.index(note_end)
                for sec_index in range(note_start_sec_index, note_end_sec_index + 1):
                    time_bins[sec_index] = True

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

                if (
                    counter == steps_per_sec * longest_allowed_space
                    or time_idx == len(time_bins) - 1
                ):
                    if chunk_end_idx - chunk_start_idx >= steps_per_sec * minimum_chunk_length:

                        chunk_start_sec = float(time_bins_sec[chunk_start_idx])
                        chunk_end_sec = float(time_bins_sec[chunk_end_idx])

                        note_chunk_start_idx = -1
                        for note_idx in range(note_idx_to_start_checking, len(instrument.notes)):
                            if (
                                note_chunk_start_idx == -1
                                and instrument.notes[note_idx].start >= chunk_start_sec
                            ):
                                note_chunk_start_idx = note_idx
                                start_note_sec = instrument.notes[note_chunk_start_idx].start
                            if instrument.notes[note_idx].start > chunk_end_sec + STEP_IN_SEC or (
                                note_idx == len(instrument.notes) - 1
                            ):  # or 뒤의 조건을 빠뜨리면 instrument의 끝과 맞닿아 있는 chunk가 포함이 안됨.
                                # pull timelilnes of notes to start of first note
                                for track_note in instrument.notes[note_chunk_start_idx:note_idx]:
                                    track_note.start -= start_note_sec
                                    track_note.end -= start_note_sec
                                new_notes_per_instrument.append(
                                    instrument.notes[note_chunk_start_idx:note_idx]
                                )
                                note_idx_to_start_checking = note_idx
                                break

            for i, notes in enumerate(new_notes_per_instrument):
                new_midi_object = pretty_midi.PrettyMIDI(
                    resolution=midi_data.resolution, initial_tempo=average_tempo
                )

                ks_list = midi_data.key_signature_changes
                ts_list = midi_data.time_signature_changes

                if len(ks_list) > MAXIMUM_CHANGE or len(ts_list) > MAXIMUM_CHANGE:
                    break

                if ks_list:  # ks 가 변화하지 않는 경우 default값으로 설정 필요
                    for ks in ks_list:
                        new_midi_object.key_signature_changes.append(ks)

                if ts_list:  # ts 가 변화하지 않는 경우 default값으로 설정 필요
                    for ts in ts_list:
                        new_midi_object.time_signature_changes.append(ts)

                new_instrument = pretty_midi.Instrument(program=instrument.program)
                new_instrument.notes = notes
                new_midi_object.instruments.append(new_instrument)
                new_midi_object.write(
                    os.path.join(
                        chunked_midi_path,
                        filename_without_extension + f"_{inst_idx}_{instrument.program}_{i}.mid",
                    )
                )


def chunk_midi(
    steps_per_sec,
    longest_allowed_space,
    minimum_chunk_length,
    midi_dataset_path,
    chunked_midi_path,
    tmp_midi_dir,
):

    midifiles = []
    for _, (dirpath, _, filenames) in enumerate(os.walk(midi_dataset_path)):
        fileExt = [".mid", ".MID", ".MIDI", ".midi"]
        for Ext in fileExt:
            tem = [os.path.join(dirpath, _) for _ in filenames if _.endswith(Ext)]
            if tem:
                midifiles += tem

    split_midi = np.array_split(np.array(midifiles), NUM_CORES)
    split_midi = [x.tolist() for x in split_midi]

    parmap.map(
        chunk_midi_map,
        split_midi,
        steps_per_sec,
        longest_allowed_space,
        minimum_chunk_length,
        chunked_midi_path,
        tmp_midi_dir,
        pm_pbar=True,
        pm_processes=NUM_CORES,
    )
