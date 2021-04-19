import argparse
import os
import pprint
import time
from pathlib import Path

from dejavu import Dejavu
from dejavu.logic.recognizer.file_recognizer import FileRecognizer

config = {
    "database": {
        "host": os.getenv("DATABASE_HOST"),
        "user": os.getenv("DATABASE_USER"),
        "password": os.getenv("DATABASE_PASSWORD"),
        "port": os.getenv("DATABASE_PORT"),
        "database": "fingerprint",
    },
    "database_type": "mysql",
}


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Audio fingerprinting using dejavu")
    parser.add_argument("--source_dir", type=str, required=True, help="해시값을 계산할 오디오 파일이 저장된 디렉토리")
    parser.add_argument("--audio_path", type=str, required=True, help="핑거프린팅 검사를 수행할 오디오 경로")
    parser.add_argument(
        "--audio_format", type=str, default="wav", nargs="+", choices=("wav", "mp3")
    )
    parser.add_argument("--num_workers", type=int, default=4)
    return parser


def main(args: argparse.Namespace) -> None:
    source_dir = str(Path(args.source_dir).expanduser())
    audio_path = str(Path(args.audio_path).expanduser())

    # 개별 파일 입력시 오류 발생 (소스 코드에 에러가 있음)
    s = time.perf_counter()
    djv = Dejavu(config)
    djv.fingerprint_directory(
        source_dir,
        extensions=[f".{fmt}" for fmt in args.audio_format],
        nprocesses=args.num_workers,
    )
    e = time.perf_counter()
    print(f"Finished fingerprinting in {e-s:.3f}s")

    result = djv.recognize(FileRecognizer, audio_path)

    print(f"From file recognized\n:{pprint.pformat(result)}\n")


if __name__ == "__main__":
    known_args, _ = get_parser().parse_known_args()
    main(known_args)
