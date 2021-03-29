from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Union

from .resources import s3


class S3Client:
    def __init__(self):
        self._client = s3.get_s3_client()

    def download_resources(
        self,
        bucket_name: str,
        s3_key: str,
        output_dir: Optional[Union[str, Path]] = None,
        download_fileobj: bool = False,
        fileobj: Optional[IO] = None,
        ignore_prefix: bool = True,
    ) -> Optional[Union[str, List[str]]]:
        """S3 버켓에서 파일/폴더를 다운로드하는 함수
        Args:
            bucket_name: `str`. 객체를 다운로드받은 S3 버켓명
            output_dir: `Optional[PathLike]`. 객체를 다운로드할 디렉토리. `download_fileobj` 비활성화시에만 필요
            s3_key: `str`. 대상 파일/폴더명을 찾기 위한 접두사
            download_fileobj: `bool`. file-like 객체 다운로드 여부
            fileobj: `Optional[IO]`. `download_fileobj` 활성화 시 콘텐츠를 저장할 객체
            ignore_prefix: `bool`. `True`일 때 접두사를 무시하고 `output_dir / filename`으로 저장
        """
        matched_objects = self.find_resource(bucket_name, s3_key)

        if download_fileobj:
            self._client.download_fileobj(
                Bucket=bucket_name, Key=matched_objects[0]["Key"], Fileobj=fileobj
            )
            return

        result = []
        output_dir = Path(output_dir)
        for obj in matched_objects:
            if not obj["Size"]:
                continue
            key = obj["Key"]
            output_path = (
                output_dir.joinpath(Path(key).name) if ignore_prefix else output_dir.joinpath(key)
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._client.download_file(bucket_name, key, str(output_path))
            result.append(str(output_path))
        return result[0] if len(result) == 1 else result

    def find_resource(self, bucket_name: str, s3_key: str) -> List[Dict[str, Any]]:
        search_result = self._client.list_objects(Bucket=bucket_name, Prefix=s3_key)
        if "Contents" not in search_result:
            raise ValueError(f"Couldn't find any objects start with {s3_key} in {bucket_name}")
        return search_result["Contents"]
