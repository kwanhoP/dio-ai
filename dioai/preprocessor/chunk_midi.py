import os

import numpy as np
import pretty_midi


def chunk_midi(
    steps_per_sec, longest_allowed_space, minimum_chunk_length, midi_dataset_path, chunked_midi_path
) -> None:
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
    # Remove drum, FX, woodblock, gunshot, ... among midi instruments
    Instrument_Not_For_Melody = [
        55,
        96,
        97,
        98,
        99,
        100,
        101,
        102,
        103,
        113,
        114,
        115,
        116,
        117,
        118,
        119,
        120,
        121,
        122,
        123,
        124,
        125,
        126,
        127,
    ]
    midifiles = []

    for i, (dirpath, _, filenames) in enumerate(os.walk(midi_dataset_path)):
        fileExt = [".mid", ".MID", ".MIDI", ".midi"]
        for Ext in fileExt:
            tem = [os.path.join(dirpath, _) for _ in filenames if _.endswith(Ext)]
            if tem:
                midifiles += tem

    for file_idx, filename in enumerate(midifiles):
        print(f"{file_idx}th file, filename: {filename}")
        try:
            midi_data = pretty_midi.PrettyMIDI(filename)
        except OSError:
            print("pretty_midi로 읽을 수 있는 형식의 mid 데이터가 아닙니다.")
            continue

        # First, remove instrument track not necessary for generating melody
        instrument_idx_to_remove = []
        for inst_idx, instrument in enumerate(midi_data.instruments):
            # if instrument is drum or not suitable for melody
            if instrument.is_drum or instrument.program in Instrument_Not_For_Melody:
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
                new_midi_object = pretty_midi.PrettyMIDI()
                new_instrument = pretty_midi.Instrument(program=instrument.program)
                new_instrument.notes = notes
                new_midi_object.instruments.append(new_instrument)
                filename_without_extension = os.path.splitext(filename.split("/")[-1])[0]
                new_midi_object.write(
                    os.path.join(
                        chunked_midi_path,
                        filename_without_extension + f"_{inst_idx}_{instrument.program}_{i}.mid",
                    )
                )
