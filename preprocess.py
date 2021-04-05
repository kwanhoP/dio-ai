# 미디 전처리 스크립트 (midi -> tfrecord)

import argparse
import os
import warnings
from pathlib import Path
from typing import List

import numpy as np

from dioai.logger import logger
from dioai.preprocessor.chunk_midi import chunk_midi
from dioai.preprocessor.extract_info import MidiExtractor, extract_midi_info
from dioai.preprocessor.utils import concat_npy, load_poza_meta, parse_midi, split_train_val_test

# Pozadataset URL
URL = "https://backoffice.pozalabs.com/api/samples"


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
        "--target_dataset",
        type=str,
        default="reddit",
        choices=["reddit", "poza"],
        help="전처리 대상이 되는 데이터셋 지정, 데이터셋에 따라 전처리 flow 차이가 존재",
    )
    parser.add_argument(
        "--source_midi_dir",
        type=str,
        required=True,
        help="전처리하고자 하는 midi파일들이 들어있는 폴더. 폴더 속 폴더 존재 가능",
    )
    parser.add_argument(
        "--bar_window_size",
        type=List,
        default=[4, 8],
        help="파싱할 미디 마디 수 지정",
    )
    parser.add_argument(
        "--num_cores",
        type=int,
        default=10,
        choices=range(1, 20),
        help="병렬 처리 시 사용할 프로세스 수 지정",
    )
    return parser


def main(args):
    # args
    STEPS_PER_SEC = args.steps_per_sec
    LONGEST_ALLOWED_SPACE = args.longest_allowed_space
    MINIMUM_CHUNK_LENGTH = args.minimum_chunk_length
    VAL_RATIO = args.val_ratio
    TEST_RATIO = args.test_ratio
    TARGET_DATASET = args.target_dataset
    STANDARD_WINDOW_SIZE = args.bar_window_size
    num_cores = args.num_cores

    # sub-path parsing
    midi_dataset_paths = Path(args.source_midi_dir)
    subset_dir = os.listdir(midi_dataset_paths)
    for subset in subset_dir:
        midi_dataset_path = midi_dataset_paths / subset

        # 이미 전처리 완료된 subset 폴더는 건너 뜀
        encode_npy_pth = midi_dataset_path / "output_npy" / "input_train.npy"
        if encode_npy_pth.exists():
            logger.info(f"------Already processed: {subset}-------")
            continue

        logger.info(f"------Start processing: {subset}-------")
        for sub_dir in [
            "chunked",
            "parsed",
            "tmp",
            "output_npy",
            "npy_tmp",
        ]:
            pth = midi_dataset_path / sub_dir
            if not pth.exists():
                os.makedirs(pth)
            if sub_dir == "chunked":
                chunk_midi_dir = pth
            elif sub_dir == "parsed":
                parsing_midi_dir = pth
            elif sub_dir == "tmp":
                tmp_midi_dir = pth
            elif sub_dir == "output_npy":
                encode_npy_dir = pth
            elif sub_dir == "npy_tmp":
                encode_tmp_dir = pth

        # chunk
        if TARGET_DATASET != "poza":
            logger.info("-----------START CHUNK-----------")
            chunk_midi(
                steps_per_sec=STEPS_PER_SEC,
                longest_allowed_space=LONGEST_ALLOWED_SPACE,
                minimum_chunk_length=MINIMUM_CHUNK_LENGTH,
                midi_dataset_path=midi_dataset_path,
                chunked_midi_path=chunk_midi_dir,
                tmp_midi_dir=tmp_midi_dir,
                num_cores=num_cores,
                dataset_name=TARGET_DATASET,
            )

        # parsing
        if TARGET_DATASET != "poza":
            logger.info("----------START PARSING----------")
            for window_size in STANDARD_WINDOW_SIZE:
                parse_midi(
                    source_dir=chunk_midi_dir,
                    num_measures=window_size,
                    shift_size=1,
                    output_dir=parsing_midi_dir,
                    num_cores=num_cores,
                )

        # extract
        if not os.listdir(parsing_midi_dir):
            print("정보를 추출할 미디 파일이 없습니다.")
            continue
        logger.info("-----------START EXTRACT---------")

        if TARGET_DATASET == "poza":
            poza_metas = load_poza_meta(URL)
            for poza_meta in poza_metas:
                metadata = MidiExtractor(
                    pth=None, keyswitch_velocity=None, default_pitch_range=None, poza_meta=poza_meta
                ).parse_poza()
                print(metadata)  # test용 추후 삭제 예정

        else:
            extract_midi_info(parsing_midi_dir, encode_tmp_dir, num_cores)

            # load, concat and save npy
            input_npy, target_npy = concat_npy(encode_tmp_dir)

            # split and save npy(results)
            splits = split_train_val_test(input_npy, target_npy, VAL_RATIO, TEST_RATIO)

            for split_name, value in splits.items():
                np.save(os.path.join(encode_npy_dir, split_name), value)

            logger.info(f"------Finish processing: {subset}-------")


if __name__ == "__main__":
    args, _ = get_parser().parse_known_args()  # noqa: F403
    warnings.filterwarnings("ignore")
    main(args)
