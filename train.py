import argparse
from pathlib import Path
from typing import Union

import sentry_sdk
import tensorflow as tf
import transformers

from dioai.config import TransformersConfig
from dioai.data.dataset import PozalabsDatasetFactory
from dioai.model import PozalabsModelFactory
from dioai.trainer import Trainer


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Train model")
    parser.add_argument("--config_path", type=str, help="전체 설정값이 저장된 JSON 파일 경로")
    return parser


def limit_tf_gpu_memory(memory_limit: int) -> None:
    # Tensorflow 의 GPU 메모리 독점을 막기 위한 GPU 메모리 점유 제한
    # https://github.com/tensorflow/tensorflow/issues/25138#issuecomment-583800729
    gpus = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_virtual_device_configuration(
        device=gpus[0],
        logical_devices=[
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)
        ],
    )


def find_latest_checkpoint(checkpoint_dir: Union[str, Path]) -> str:
    def _sort_by_checkpoint(_p: Path) -> int:
        return int(_p.stem.split("-")[-1])

    # 체크포인트는 `checkpoint-{step}`으로 저장됨
    sorted_checkpoints = sorted(
        (
            dir_name
            for dir_name in Path(checkpoint_dir).iterdir()
            if dir_name.is_dir() and dir_name.stem.startswith("checkpoint")
        ),
        key=_sort_by_checkpoint,
    )
    return str(sorted_checkpoints[-1])


def load_config(config_path: Union[str, Path]) -> TransformersConfig:
    config = TransformersConfig.from_json(config_path)
    config.save()
    return config


def main(args):
    sentry_sdk.init(
        "https://11345898b114459fb6eb068986b66eea@o226139.ingest.sentry.io/5690046",
        traces_sample_rate=1.0,
    )

    config = load_config(Path(args.config_path).expanduser())

    limit_tf_gpu_memory(config.tf_gpu_memory_limit)

    dataset_factory = PozalabsDatasetFactory()
    model_factory = PozalabsModelFactory()

    trainer: transformers.Trainer = Trainer(
        model=model_factory.create(config.model_name, config.model),
        args=config.training,
        train_dataset=dataset_factory.create(config=config, split=config.train_split),
        eval_dataset=dataset_factory.create(
            config=config, split=config.eval_split, training=False, shuffle=False
        ),
    )
    trainer.train(
        resume_from_checkpoint=(
            find_latest_checkpoint(config.training.output_dir) if config.resume_training else None
        )
    )


if __name__ == "__main__":
    known_args, _ = get_parser().parse_known_args()
    main(known_args)
