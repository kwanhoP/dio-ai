# poza 미디 데이터셋 다운로드를 위한 AWS S3 다운로더
# ref) https://github.com/POZAlabs/dio/blob/master/utils/client/s3.py

import argparse
import functools
from pathlib import Path
from typing import IO, List, Optional, Union

import boto3
from botocore.exceptions import ClientError

PathLike = Union[str, Path]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("S3에서 미디 파일을 다운로드 합니다.")
    parser.add_argument(
        "--bucket",
        type=str,
        default="dio-samples",
        help="다운로드 할 s3 버킷명",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="다운로드 파일 저장 경로",
        required=True,
    )
    parser.add_argument(
        "--aws_access_key_id",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--aws_secret_access_key",
        type=str,
        required=True,
    )
    return parser


def try_request(func):
    @functools.wraps(func)
    def decorator(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except ClientError:
            return False
        return True

    return decorator


class S3Client:
    def __init__(self, id, key):
        self.client = boto3.client(
            "s3",
            aws_access_key_id=id,
            aws_secret_access_key=key,
        )

    @try_request
    def download_file_or_directory(
        self,
        bucket_name: str,
        s3_key: str,
        output_path: Optional[PathLike] = None,
        matched_object: List = None,
        fileobj: Optional[IO] = None,
    ):
        if fileobj is not None:
            self.client.download_fileobj(Bucket=bucket_name, Key=s3_key, Fileobj=fileobj)
            return

        if output_path is None:
            raise ValueError("`output_path` should not be None")

        output_path = Path(output_path)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        if matched_object:
            matched_objects = matched_object
        else:
            matched_objects = self.client.list_objects(Bucket=bucket_name, Prefix=s3_key)[
                "Contents"
            ]
        if len(matched_objects) > 1:
            for obj in matched_objects:
                obj_key = obj["Key"]
                if obj_key.endswith("mid"):
                    self.download_file_or_directory(
                        bucket_name=bucket_name,
                        s3_key=obj_key,
                        output_path=str(output_path.joinpath(*Path(obj_key).parts[1:])),
                    )
        else:
            self.client.download_file(bucket_name, s3_key, str(output_path))


def download_midi_from_s3(S3Client, bucket: str, save_dir: Path) -> None:
    """
    S3Client: S3Client instance
    bucket: 다운로드 할 s3 버킷명
    save_dir: 다운로드 파일 저장 경로
    NextContinuationToken으로 다음 객체를 계속 받아와 전체 다운로드
    """
    next_object = ""
    while next_object is not None:
        if next_object == "":
            object = S3Client.client.list_objects_v2(Bucket=bucket, Prefix="")
        else:
            object = S3Client.client.list_objects_v2(
                Bucket=bucket, Prefix="", ContinuationToken=next_object
            )
        S3Client.download_file_or_directory(bucket, "", save_dir, object["Contents"])
        next_object = object.get("NextContinuationToken")


if __name__ == "__main__":
    args, _ = get_parser().parse_known_args()  # noqa: F403
    bucket = args.bucket
    save_dir = args.save_dir
    aws_access_key_id = args.aws_access_key_id
    aws_secret_access_key = args.aws_secret_access_key

    s3 = S3Client(aws_access_key_id, aws_secret_access_key)
    download_midi_from_s3(s3, bucket, save_dir)
