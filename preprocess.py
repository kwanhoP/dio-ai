import argparse
import os
from multiprocessing import cpu_count
from pathlib import Path
from typing import Optional

from dioai.logger import logger
from dioai.preprocessor import ChunkMidiArguments, ParseMidiArguments
from dioai.preprocessor.pipeline import PREPROCESS_STEPS, PreprocessPipeline

ALLOWED_DATASETS = ("pozalabs", "pozalabs2", "reddit")
NO_PREPARATION_BEFORE_ENCODING = ("pozalabs",)
PRESERVE_CHORD_TRACK = ("pozalabs2",)
ARGUMENT_GROUP_ADD_FUNC_NAME_FORMAT = "add_{}_argument_group"
DATASET_NAME_ENV = "DATASET_NAME"


def get_root_parser() -> argparse.ArgumentParser:
    # https://docs.python.org/ko/3/library/argparse.html#parents
    root_parser = argparse.ArgumentParser("데이터셋 전처리 스크립트", add_help=True)
    root_parser.add_argument("--root_dir", type=str, required=True, help="전처리 대상 데이터가 저장된 루트 디렉토리")
    root_parser.add_argument(
        "--num_cores", type=int, help="병렬 처리시 사용할 프로세스 개수", default=max(1, cpu_count() - 4)
    )
    root_parser.add_argument("--augment", action="store_true", help="데이터 어그멘테이션 수행 여부")
    root_parser.add_argument("--steps", nargs="+", default=PREPROCESS_STEPS, help="전처리 수행할 단계")
    return root_parser


def add_midi_chunking_argument_group(root_parser: argparse.ArgumentParser) -> None:
    group = root_parser.add_argument_group("데이터셋 청크")
    group.add_argument(
        "--steps_per_sec",
        type=int,
        default=10,
        choices=(10, 100),
        help="각 청크의 길이를 계산할 때 사용되는 초당 스텝 (1/step_per_sec으로 계산)",
    )
    group.add_argument(
        "--longest_allowed_space",
        type=int,
        default=2,
        help="하나의 청크 내에서 허용되는 가장 긴 공백 (쉼표) 길이 (초)",
    )
    group.add_argument("--minimum_chunk_length", type=int, default=8, help="청크의 최소 단위 (초)")


def add_midi_parsing_argument_group(root_parser: argparse.ArgumentParser) -> None:
    group = root_parser.add_argument_group("데이터셋 마디 단위 파싱")
    group.add_argument(
        "--bar_window_size",
        type=int,
        nargs="+",
        default=[4, 8],
        help="파싱할 마디 수",
    )
    group.add_argument("--shift_size", type=int, default=1, help="마디 파싱 시 다음 마디 이동 크기 (마디)")


def add_reddit_argument_group(root_parser: argparse.ArgumentParser) -> None:
    add_midi_chunking_argument_group(root_parser)
    add_midi_parsing_argument_group(root_parser)


def add_pozalabs2_argument_group(root_parser: argparse.ArgumentParser) -> None:
    add_midi_chunking_argument_group(root_parser)
    add_midi_parsing_argument_group(root_parser)
    group = root_parser.add_argument_group("포자랩스2 데이터 전처리 관련 인자")
    group.add_argument("--chord_progression_csv_path", type=str, help="코드 진행이 저장된 CSV 파일 경로")


def add_pozalabs_argument_group(root_parser: argparse.ArgumentParser) -> None:
    group = root_parser.add_argument_group("포자랩스 데이터 전처리 관련 인자")
    group.add_argument(
        "--backoffice_api_url",
        type=str,
        default="https://backoffice.pozalabs.com",
        help="백오피스 API URL",
    )
    group.add_argument(
        "--update_date",
        type=str,
        default="1999-01-01",
        help="업데이트 할 최신 날짜"
    )


def get_parser(dataset_name: str) -> argparse.ArgumentParser:
    root_parser = get_root_parser()
    argument_group_add_func_name = ARGUMENT_GROUP_ADD_FUNC_NAME_FORMAT.format(dataset_name)
    argument_group_add_func = globals()[argument_group_add_func_name]
    argument_group_add_func(root_parser)
    return root_parser


def prepare_path(_path) -> Path:
    return Path(_path).expanduser()


def get_dataset_name() -> Optional[str]:
    return os.getenv("DATASET_NAME")


def main(args: argparse.Namespace) -> None:
    dataset_name = get_dataset_name()
    root_dir = prepare_path(args.root_dir)
    pipeline = PreprocessPipeline(dataset_name)
    preprocess_steps = args.steps
    logger.info(f"Data: {dataset_name} | Preprocess Steps: {preprocess_steps}")

    if dataset_name in NO_PREPARATION_BEFORE_ENCODING:
        dataset_extra_args = dict(
            backoffice_api_url=args.backoffice_api_url,
            update_date=args.update_date
        )
    else:
        dataset_extra_args = dict(
            chunk_midi_arguments=ChunkMidiArguments(
                steps_per_sec=args.steps_per_sec,
                longest_allowed_space=args.longest_allowed_space,
                minimum_chunk_length=args.minimum_chunk_length,
                preserve_chord_track=dataset_name in PRESERVE_CHORD_TRACK,
                preserve_channel=dataset_name in PRESERVE_CHORD_TRACK,
            ),
            parse_midi_arguments=ParseMidiArguments(
                bar_window_size=args.bar_window_size,
                shift_size=args.shift_size,
                preserve_channel=dataset_name in PRESERVE_CHORD_TRACK,
            ),
        )
        if dataset_name in PRESERVE_CHORD_TRACK:
            dataset_extra_args["chord_progression_csv_path"] = Path(
                args.chord_progression_csv_path
            ).expanduser()

    pipeline(
        root_dir=root_dir,
        num_cores=args.num_cores,
        augment=args.augment,
        preprocess_steps=preprocess_steps,
        **dataset_extra_args,
    )


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    # `DATASET_NAME`을 환경 변수로 전달해야 합니다.
    # e.g.) DATASET_NAME=pozalabs python3 preprocess_v2.py ...
    _dataset_name = get_dataset_name()
    if _dataset_name is None:
        raise RuntimeError("You must set DATASET_NAME as environment variable")

    if _dataset_name not in ALLOWED_DATASETS:
        raise RuntimeError(f"`DATASET_NAME` should be one of {ALLOWED_DATASETS}")

    parser = get_parser(_dataset_name)
    known_args, _ = parser.parse_known_args()
    main(known_args)