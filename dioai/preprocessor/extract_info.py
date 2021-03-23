import os
from pathlib import Path
from typing import Dict, List

import mido
import numpy as np
import parmap
from tqdm import tqdm

from .constants import (
    DEFAULT_BPM,
    DEFAULT_KEY,
    DEFAULT_TS,
    KEY_MAP,
    NUM_CORES,
    PITCH_RANGE_MAP,
    TIME_SIG_MAP,
    UNKNOWN,
)
from .container import MidiInfo
from .encoder import encode_midi
from .utils import (
    encode_meta_info,
    get_bpm,
    get_inst_from_midi,
    get_key_chord_type,
    get_meta_message,
    get_num_measures_from_midi,
    get_pitch_range,
    get_time_signature,
)


class MidiExtractor:
    """미디 정보를 추출합니다.

    파싱되는 정보:
        # meta
        - bpm
        - audio_key
        - time_signature
        - pitch_range
        - num_measure
        - inst

        # note
    """

    def __init__(
        self, pth: str, keyswitch_velocity: int, default_pitch_range: str, poza_meta: Dict
    ):
        """

        Args:
            pth: `str`. 인코딩 할 미디 path(chunked and parsing)
            keyswitch_velocity: `int`. pitch range 검사에서 제외할 keyswitch velocity
            default_pitch_range: `str`. 모든 노트의 velocity 가 keyswitch velocity 라서
                        pitch range를 검사할 수 없을 경우 사용할 기본 pitch range

        """
        if pth:
            self._midi = mido.MidiFile(pth)
            self.note_seq = encode_midi(pth)
        self.keyswitch_velocity = keyswitch_velocity
        self.default_pitch_range = default_pitch_range
        self.path = pth
        self.poza_meta = poza_meta

    def parse(self) -> MidiInfo:
        meta_track = self._midi.tracks[0]
        key = get_key_chord_type(get_meta_message(meta_track, "key_signature"))

        midi_info = MidiInfo(
            bpm=get_bpm(get_meta_message(meta_track, "set_tempo"), poza_bpm=None),
            audio_key=key,
            time_signature=get_time_signature(get_meta_message(meta_track, "time_signature")),
            pitch_range=get_pitch_range(self._midi, self.keyswitch_velocity),
            num_measure=get_num_measures_from_midi(self.path),
            inst=get_inst_from_midi(self.path),
            note_seq=self.note_seq,
        )

        return midi_info

    def parse_poza(self) -> MidiInfo:
        midi_info = MidiInfo(
            bpm=get_bpm(meta_message=None, poza_bpm=self.poza_meta["bpm"]),
            audio_key=KEY_MAP[self.poza_meta["audio_key"] + self.poza_meta["chord_type"]],
            time_signature=TIME_SIG_MAP[self.poza_meta["time_signature"]],
            pitch_range=PITCH_RANGE_MAP[self.poza_meta["pitch_range"]],
            num_measure=self.poza_meta["num_measures"],
            inst=self.poza_meta["inst"],
            note_seq=None,
        )
        return midi_info


def extract_midi_info_map(chunked_midi: List, encode_tmp_dir: Path) -> None:
    for i, midi_file in tqdm(enumerate(chunked_midi)):
        metadata = MidiExtractor(
            pth=midi_file, keyswitch_velocity=1, default_pitch_range="mid", poza_meta=None
        ).parse()
        if (
            (metadata.bpm == DEFAULT_BPM)
            and (metadata.audio_key == DEFAULT_KEY)
            and (metadata.time_signature == DEFAULT_TS)
        ):
            metadata.bpm = UNKNOWN
            metadata.audio_key = UNKNOWN
            metadata.time_signature = UNKNOWN
        meta = encode_meta_info(metadata)
        if meta:
            input_npy = np.array(np.array(meta), dtype=object)
            target_npy = np.array(np.array(metadata.note_seq), dtype=object)
            np.save(os.path.join(encode_tmp_dir, f"input_{i}"), input_npy)
            np.save(os.path.join(encode_tmp_dir, f"target_{i}"), target_npy)
        else:
            continue


def extract_midi_info(parsing_midi_pth: Path, encode_tmp_dir: Path) -> None:
    midifiles = []

    for _, (dirpath, _, filenames) in enumerate(os.walk(parsing_midi_pth)):
        midi_extensions = [".mid", ".MID", ".MIDI", ".midi"]
        for ext in midi_extensions:
            tem = [os.path.join(dirpath, _) for _ in filenames if _.endswith(ext)]
            if tem:
                midifiles += tem

    split_midi = np.array_split(np.array(midifiles), NUM_CORES)
    split_midi = [x.tolist() for x in split_midi]
    parmap.map(
        extract_midi_info_map,
        split_midi,
        encode_tmp_dir,
        pm_pbar=True,
        pm_processes=NUM_CORES,
    )
