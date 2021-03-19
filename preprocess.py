# 미디 전처리 스크립트 (midi -> tfrecord)

import argparse
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from dioai.preprocessor.chunk_midi import chunk_midi
from dioai.preprocessor.constants import DEFAULT_BPM, DEFAULT_KEY, DEFAULT_TS, META_LEN, UNKNOWN
from dioai.preprocessor.extract_info import MidiExtractor
from dioai.preprocessor.utils import encode_meta_info, parse_midi, split_train_val_test

# 4마디, 8마디 단위로 데이터화
STANDARD_WINDOW_SIZE = [4, 8]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("미디 전처리 후 Numpy array 형식으로 변환합니다.")
    parser.add_argument(
        "--steps_per_sec",
        type=int,
        default=10,
        choices=[10, 100],
        help="각 Chunk의 길이를 계산할 때 1./steps_per_sec 단위로 계산.",
    )
    parser.add_argument(
        "--longest_allowed_space",
        type=int,
        default=2,
        help="하나의 chunk안에서 제일 긴 space는 longest_allowed_space초를 넘을 수 없다",
    )
    parser.add_argument(
        "--minimum_chunk_length",
        type=int,
        default=8,
        help="길이가 minimum_chunk_length초 미만인 chunk는 누락된다",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="validation set 비율",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="test set 비율",
    )
    parser.add_argument(
        "--source_midi_dir",
        type=str,
        required=True,
        help="전처리하고자 하는 midi파일들이 들어있는 폴더. 폴더 속 폴더 존재 가능",
    )
    parser.add_argument(
        "--chunk_midi_dir",
        type=str,
        required=True,
        help="전처리 후 만들어진 각 chunk 는 하나의 midi파일로 저장되는데, 이러한 midi 파일들이 저장되는 폴더",
    )
    parser.add_argument(
        "--tmp_midi_dir",
        type=str,
        required=True,
        help="미디 파일의 Tempo가 달라지는 경우 노트가 밀리는 현상 해결을 위해 평균 Tempo값으로 고정되어 저장되는 폴더",
    )
    parser.add_argument(
        "--after_chunked",
        type=bool,
        default=False,
        help="chunked 이후 과정만 할 때, --chunk_midi_dir을 대상으로 인코딩 진행",
    )
    parser.add_argument(
        "--window_chunk_dir",
        type=str,
        default=False,
        help="chunked된 미디 파일을 마디 길이에 맞게 Augment 이후 저장되는 폴더",
    )
    parser.add_argument(
        "--encode_npy_dir",
        type=str,
        default=True,
        help="인코딩 된 미디 데이터를 npy 형식으로 저장하는 폴더",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="GPT",
        help="모델 구조에 따른 전처리 형식 지정 GPT: (meta + note_seq) 합쳐서 하나의 시퀀스로 학습",
    )
    return parser


def main(args):
    # args
    STEPS_PER_SEC = args.steps_per_sec
    LONGEST_ALLOWED_SPACE = args.longest_allowed_space
    MINIMUM_CHUNK_LENGTH = args.minimum_chunk_length
    VAL_RATIO = args.val_ratio
    TEST_RATIO = args.test_ratio
    MODEL = args.model
    after_chunked = args.after_chunked

    # sub-path parsing
    midi_dataset_paths = args.source_midi_dir
    subset_dir = os.listdir(midi_dataset_paths)
    for subset in subset_dir:
        print(f"------Start processing: {subset}-------")
        midi_dataset_path = os.path.join(midi_dataset_paths, subset)
        # 이미 전처리 완료된 subset 폴더는 건너 뜀
        if os.path.exists(os.path.join(midi_dataset_path, args.encode_npy_dir, "input.npy")):
            continue

        chunked_midi_path = os.path.join(midi_dataset_path, args.chunk_midi_dir)
        if not os.path.exists(chunked_midi_path):
            os.makedirs(chunked_midi_path)

        window_chunked_dir = os.path.join(midi_dataset_path, args.window_chunk_dir)
        if not os.path.exists(window_chunked_dir):
            os.makedirs(window_chunked_dir)

        tmp_midi_dir = os.path.join(midi_dataset_path, args.tmp_midi_dir)
        if not os.path.exists(tmp_midi_dir):
            os.makedirs(tmp_midi_dir)

        encode_npy_dir = os.path.join(midi_dataset_path, args.encode_npy_dir)
        if not os.path.exists(encode_npy_dir):
            os.makedirs(encode_npy_dir)
        # chunk
        if not after_chunked:
            print("---------------------------------")
            print("-----------START CHUNK-----------")
            print("---------------------------------")
            chunk_midi(
                steps_per_sec=STEPS_PER_SEC,
                longest_allowed_space=LONGEST_ALLOWED_SPACE,
                minimum_chunk_length=MINIMUM_CHUNK_LENGTH,
                midi_dataset_path=midi_dataset_path,
                chunked_midi_path=chunked_midi_path,
                tmp_midi_dir=tmp_midi_dir,
            )

        parsing_midi_pth = Path(window_chunked_dir)
        print("---------------------------------")
        print("----------START PARSING----------")
        print("---------------------------------")
        for window_size in STANDARD_WINDOW_SIZE:
            parse_midi(
                midi_path=chunked_midi_path,
                num_measures=window_size,
                shift_size=1,
                parsing_midi_pth=parsing_midi_pth,
            )

        # extract & encode
        chunked_midi = []

        for _, (dirpath, _, filenames) in enumerate(os.walk(parsing_midi_pth)):
            file_ext = [".mid", ".MID", ".MIDI", ".midi"]
            for Ext in file_ext:
                tem = [os.path.join(dirpath, _) for _ in filenames if _.endswith(Ext)]
                if tem:
                    chunked_midi += tem

        input_meta = []
        target_note = []
        print("---------------------------------")
        print("-----------START EXTRACT---------")
        print("---------------------------------")
        for midi_file in tqdm(chunked_midi):
            metadata = MidiExtractor(
                pth=midi_file, keyswitch_velocity=1, default_pitch_range="mid"
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
                input_meta.append(np.array(meta))
                target_note.append(np.array(metadata.note_seq))
            else:
                continue

        input_npy = np.array(input_meta, dtype=object)
        target_npy = np.array(target_note, dtype=object)
        print(input_npy.shape, target_npy.shape)
        if MODEL == "GPT":
            target_npy = target_npy + META_LEN

        # split data
        splits = split_train_val_test(input_npy, target_npy, VAL_RATIO, TEST_RATIO)

        for split_name, value in splits.items():
            np.save(os.path.join(encode_npy_dir, split_name), value)

        print(f"------Finish processing: {subset}-------")


if __name__ == "__main__":
    args, _ = get_parser().parse_known_args()  # noqa: F403
    main(args)
