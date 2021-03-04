# 미디 전처리 스크립트 (midi -> tfrecord)

import argparse


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("미디 전처리 후 tfrecord 형식으로 변환합니다.")
    return parser


if __name__ == "__main__":
    args, _ = get_parser().parse_known_args()  # noqa: F403
