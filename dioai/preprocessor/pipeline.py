import inspect
import time
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, Union

from dioai.logger import logger

from .encoder import MetaEncoderFactory, MidiPerformanceEncoder
from .parser import MetaParserFactory
from .preprocessor import PreprocessorFactory


class PreprocessPipeline:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def __call__(
        self,
        root_dir: Union[str, Path],
        num_cores: int = max(4, cpu_count() - 2),
        augment: bool = False,
        *args,
        **kwargs,
    ):
        meta_parser = MetaParserFactory().create(
            self.dataset_name,
            # meta_csv_path: pozalabs2 데이터셋을 위한 인자
            meta_csv_path=Path(root_dir).joinpath("meta.csv"),
        )
        meta_encoder = MetaEncoderFactory().create(self.dataset_name)
        preprocessor = PreprocessorFactory().create(
            self.dataset_name,
            meta_parser=meta_parser,
            meta_encoder=meta_encoder,
            note_sequence_encoder=MidiPerformanceEncoder(),
            *args,
            **kwargs,
        )
        logger.info(f"[{self.dataset_name}] Initialized preprocessor")
        logger.info("Start preprocessing")
        start_time = time.perf_counter()
        preprocessor.preprocess(
            root_dir=root_dir,
            num_cores=num_cores,
            augment=augment,
            **inject_args(preprocessor.preprocess, **kwargs),
        )
        end_time = time.perf_counter()
        logger.info(f"Finished preprocessing in {end_time - start_time:.3f}s")


def inject_args(func, **kwargs) -> Dict[str, Any]:
    args = [arg for arg in inspect.getfullargspec(func).args]
    return {key: value for key, value in kwargs.items() if key in args}
