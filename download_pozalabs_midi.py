import argparse
import concurrent.futures
from pathlib import Path

from dioai.logger import logger
from dioai.preprocessor import utils
from dioai.utils.sdk.s3 import S3Client


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("S3에서 미디 파일을 다운로드 합니다.")
    parser.add_argument("--backoffice_api_url", type=str, default="https://backoffice.pozalabs.com")
    parser.add_argument("--bucket_name", type=str, default="dio-samples", help="다운로드 할 S3 버킷명")
    parser.add_argument(
        "--midi_files_prefix", type=str, default="midi_files", help="미디 파일이 저장된 S3 Prefix"
    )
    parser.add_argument("--num_workers", type=int, default=10, help="다운로드 시 사용할 워커 수")
    parser.add_argument("--output_dir", type=str, help="다운로드 파일 저장 경로", required=True)
    return parser


# TODO: 2021.03.27
# 매번 전체 샘플을 다운로드하는 것은 비효율적이므로 이전 다운로드 일자와 샘플이 S3에서 변경된 날짜를 비교하여 변경된 파일만 다운로드하도록 변경
def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser()
    pozalabs_samples = utils.load_poza_meta(args.backoffice_api_url + "/api/samples", per_page=2000)
    logger.info(f"Fetched {len(pozalabs_samples)} samples from backoffice")

    s3_client = S3Client()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures_to_key = dict()
        for sample in pozalabs_samples:
            s3_key = "/".join((args.midi_files_prefix, sample["path"]))
            future = executor.submit(
                s3_client.download_resources,
                bucket_name=args.bucket_name,
                s3_key=s3_key,
                output_dir=output_dir,
                ignore_prefix=True,
            )
            futures_to_key[future] = sample["id"]

        for future in concurrent.futures.as_completed(futures_to_key):
            sample_id = futures_to_key[future]
            try:
                future.result()
                logger.info(f"Downloaded midi file of {sample_id}")
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to download midi file [{sample_id}] from s3 due to {exc}"
                )


if __name__ == "__main__":
    known_args, _ = get_parser().parse_known_args()
    main(known_args)
