# 미디 전처리 스크립트 (midi -> tfrecord)

import argparse
import os

from dioai.preprocessor.chunk_midi import chunk_midi
from dioai.preprocessor.extract_info import MetaExtractor


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
    return parser


def main(args):
    # args
    STEPS_PER_SEC = args.steps_per_sec
    LONGEST_ALLOWED_SPACE = args.longest_allowed_space
    MINIMUM_CHUNK_LENGTH = args.minimum_chunk_length

    midi_dataset_path = args.source_midi_dir
    chunked_midi_path = args.chunk_midi_dir
    after_chunked = args.after_chunked
    tmp_midi_dir = args.tmp_midi_dir
    # chunk
    if not after_chunked:
        chunk_midi(
            steps_per_sec=STEPS_PER_SEC,
            longest_allowed_space=LONGEST_ALLOWED_SPACE,
            minimum_chunk_length=MINIMUM_CHUNK_LENGTH,
            midi_dataset_path=midi_dataset_path,
            chunked_midi_path=chunked_midi_path,
            tmp_midi_dir=tmp_midi_dir,
        )

    # extract & encode
    chunked_midi = []

    for _, (dirpath, _, filenames) in enumerate(os.walk(chunked_midi_path)):
        fileExt = [".mid", ".MID", ".MIDI", ".midi"]
        for Ext in fileExt:
            tem = [os.path.join(dirpath, _) for _ in filenames if _.endswith(Ext)]
            if tem:
                chunked_midi += tem

    for midi_file in chunked_midi:
        metadata = MetaExtractor(
            pth=midi_file, keyswitch_velocity=1, default_pitch_range="mid"
        ).parse()
        print(metadata)


if __name__ == "__main__":
    args, _ = get_parser().parse_known_args()  # noqa: F403
    main(args)
