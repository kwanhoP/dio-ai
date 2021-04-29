import argparse
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import mido
import numpy as np
import pandas as pd
import tqdm

from dioai.preprocessor.utils import constants, utils
from dioai.preprocessor.utils.chord import ChordParser
from dioai.preprocessor.utils.exceptions import InvalidMidiError

CHORD_INFO_COLS = ("filename", "chord_progression", "chord_progression_hash")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("미디의 코드 진행을 추출합니다.")
    parser.add_argument(
        "--source_path",
        type=str,
        help="pozalabs2 데이터셋의 parsed 폴더 위치",
    )
    parser.add_argument(
        "--num_cores",
        type=int,
        default=max(1, cpu_count() - 4),
        help="병렬 처리시 사용할 프로세스 개수",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=200,
        help="병렬 처리시 한 프로세스당 처리할 미디 파일의 개수",
    )
    return parser


def get_chord_progression(midi_path: Union[str, Path]) -> Tuple[str, Optional[List[str]], Optional[str]]:
    filename = str(Path(midi_path).stem)
    mido_obj = mido.MidiFile(midi_path)
    try:
        chord_parser = ChordParser(midi=mido_obj).parse()
        chord_progression = chord_parser.chord_progressions[0]
        chord_progression_hash = utils.get_chord_progression_md5(chord_progression)
        return filename, chord_progression, chord_progression_hash
    except InvalidMidiError:
        return filename, None, None


def save_as_npy_file(_chord_progression_list: List[List[str]], npy_path: Union[str, Path]) -> None:
    chord_progression_array = np.array(_chord_progression_list)
    np.save(npy_path, chord_progression_array)


def save_as_csv_file(_filename_to_chord_dict: Dict[str, List[str]], csv_path: Union[str, Path]) -> None:
    df = pd.DataFrame(_filename_to_chord_dict, columns=CHORD_INFO_COLS)
    df.to_csv(csv_path, index=False)


def main(args: argparse.Namespace) -> None:
    source_path = Path(args.source_path).expanduser()
    num_cores = args.num_cores
    chunk_size = args.chunk_size

    midi_paths = [
        str(filename)
        for filename in Path(source_path).rglob("**/*")
        if filename.suffix in constants.MIDI_EXTENSIONS
    ]

    chord_progression_list = []
    filename_to_chord_dict = {col: [] for col in CHORD_INFO_COLS}
    with Pool(num_cores) as pool:
        with tqdm.tqdm(total=len(midi_paths), desc="Extracting chords") as pbar:
            for filename, chord_progression, chord_progression_hash in pool.imap_unordered(
                get_chord_progression, midi_paths, chunksize=chunk_size
            ):
                if chord_progression:
                    chord_progression_list.append(chord_progression)

                    chord_progression = ",".join(chord_progression)
                    chord_infos = (filename, chord_progression, chord_progression_hash)
                    for col, chord_info in zip(CHORD_INFO_COLS, chord_infos):
                        filename_to_chord_dict[col].append(chord_info)
                pbar.update()

    npy_path = source_path.parent.joinpath("chord_progression.npy")
    save_as_npy_file(chord_progression_list, npy_path)
    csv_path = source_path.parent.joinpath("chord_progression.csv")
    save_as_csv_file(filename_to_chord_dict, csv_path)


if __name__ == "__main__":
    known_args, _ = get_parser().parse_known_args()
    main(known_args)
