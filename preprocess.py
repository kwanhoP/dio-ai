# 미디 전처리 스크립트 (midi -> tfrecord)

import argparse

from dioai.preprocessor.chunk_midi import chunk_midi


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
    return parser


def main(args):
    # how to devide one second. should be 10 or 100 or 1000
    STEPS_PER_SEC = args.steps_per_sec
    # longest allowed space in a chunk, in second
    LONGEST_ALLOWED_SPACE = args.longest_allowed_space
    # minimum possible midi chunk, in second
    MINIMUM_CHUNK_LENGTH = args.minimum_chunk_length

    midi_dataset_path = args.source_midi_dir
    chunked_midi_path = args.chunk_midi_dir

    return chunk_midi(
        steps_per_sec=STEPS_PER_SEC,
        longest_allowed_space=LONGEST_ALLOWED_SPACE,
        minimum_chunk_length=MINIMUM_CHUNK_LENGTH,
        midi_dataset_path=midi_dataset_path,
        chunked_midi_path=chunked_midi_path,
    )


if __name__ == "__main__":
    args, _ = get_parser().parse_known_args()  # noqa: F403
    main(args)
