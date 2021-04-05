import time
from multiprocessing import cpu_count
from pathlib import Path
from typing import Union

from dioai.logger import logger

from .encoder import MetaEncoderFactory, MidiPerformanceEncoder
from .parser import MetaParserFactory
from .preprocessor import PreprocessorFactory


class PreprocessPipeline:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def __call__(
        self,
        source_dir: Union[str, Path],
        num_cores: int = max(4, cpu_count() - 2),
        *args,
        **kwargs,
    ):
        meta_parser = MetaParserFactory().create(self.dataset_name)
        meta_encoder = MetaEncoderFactory().create(self.dataset_name)
        preprocessor = PreprocessorFactory().create(
            self.dataset_name,
            meta_parser=meta_parser,
            meta_encoder=meta_encoder,
            note_sequence_encoder=MidiPerformanceEncoder(),
        )
        logger.info(f"[{self.dataset_name}] Initialized preprocessor")
        logger.info("Start preprocessing")
        start_time = time.perf_counter()
        preprocessor.preprocess(
            source_dir=source_dir,
            num_cores=num_cores,
            *args,
            **kwargs,
        )
        end_time = time.perf_counter()
        logger.info(f"Finished preprocessing in {end_time - start_time:.3f}s")
