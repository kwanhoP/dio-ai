import time
from multiprocessing import cpu_count
from pathlib import Path
from typing import List, Union

from dioai.logger import logger

from ..utils import dependency
from .encoder import CpEncoder, MetaEncoderFactory, MidiPerformanceEncoder, RemiEncoder
from .parser import MetaParserFactory
from .preprocessor import PreprocessorFactory

PREPROCESS_STEPS = ("chunk", "parse", "encode")


class PreprocessPipeline:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def __call__(
        self,
        root_dir: Union[str, Path],
        preprocess_steps: List[str],
        num_cores: int = max(4, cpu_count() - 2),
        augment: bool = False,
        remi_resolution: int = 32,
        encoder_name: str = "midi",
        *args,
        **kwargs,
    ):
        ENCODER_MAP = {
            "remi": RemiEncoder(remi_resolution),
            "midi": MidiPerformanceEncoder(),
            "cp": CpEncoder(remi_resolution),
        }

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
            note_sequence_encoder=ENCODER_MAP[encoder_name],
            encoder_name=encoder_name,
            **kwargs,
        )
        logger.info(f"[{self.dataset_name}] Initialized preprocessor")
        logger.info("Start preprocessing")
        start_time = time.perf_counter()
        preprocessor.preprocess(
            root_dir=root_dir,
            num_cores=num_cores,
            augment=augment,
            preprocess_steps=preprocess_steps,
            **dependency.inject_args(preprocessor.preprocess, **kwargs),
        )
        end_time = time.perf_counter()
        logger.info(f"Finished preprocessing in {end_time - start_time:.3f}s")
