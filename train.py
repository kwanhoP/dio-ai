import argparse
import os
from pathlib import Path
from typing import Union

import sentry_sdk
import tensorflow as tf
import transformers
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin

import wandb
from dioai.config import PytorchlightConfig, TransformersConfig
from dioai.data.dataset import PozalabsDatasetFactory
from dioai.model import ConditionalRelativeTransformer, ModelType, PozalabsModelFactory
from dioai.trainer import Trainer_hf


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Train model")
    parser.add_argument("--config_path", type=str, help="전체 설정값이 저장된 JSON 파일 경로")
    parser.add_argument("--model_type", type=str, help="모델 구현체 type", choices=["hf", "pl"])
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


def load_config(config_path: Union[str, Path], model_type: str) -> TransformersConfig:
    if model_type == ModelType.HuggingFace.value:
        config = TransformersConfig.from_file(config_path)
        config.fine_tune_ckpt = (
            find_latest_checkpoint(config.output_root_dir) if config.resume_training else None
        )
        config.save()
    elif model_type == ModelType.PytorchLightning.value:
        config = PytorchlightConfig.from_file(config_path)
    return config


def main_hf(args):
    sentry_sdk.init(
        "https://11345898b114459fb6eb068986b66eea@o226139.ingest.sentry.io/5690046",
        traces_sample_rate=1.0,
    )

    config = load_config(Path(args.config_path).expanduser(), args.model_type)

    limit_tf_gpu_memory(config.tf_gpu_memory_limit)

    dataset_factory = PozalabsDatasetFactory()
    model_factory = PozalabsModelFactory()

    trainer: transformers.Trainer = Trainer_hf(
        model=model_factory.create(config.model_name, config.model),
        args=config.training,
        train_dataset=dataset_factory.create(
            config=config, split=config.train_split, switch=config.switch
        ),
        eval_dataset=(
            dataset_factory.create(
                config=config,
                split=config.eval_split,
                switch=config.switch,
                training=False,
                shuffle=False,
            )
            if config.training.evaluation_strategy != transformers.EvaluationStrategy.NO
            else None
        ),
        use_cosine_annealing=config.use_cosine_annealing,
        num_cycles=config.num_cycles,
    )
    trainer.train(resume_from_checkpoint=config.fine_tune_ckpt)


def main_pl(args):
    os.environ["PYTHONWARNINGS"] = "ignore:semaphore_tracker:UserWarning"

    model_factory = PozalabsModelFactory()
    config = load_config(args.config_path, args.model_type)
    wandb_logger = WandbLogger(name=config.wandb_name, project=config.wandb_project)

    trainer = Trainer(
        logger=wandb_logger,
        gpus=config.n_gpu,
        fast_dev_run=False,
        accelerator="ddp",
        plugins=DDPPlugin(find_unused_parameters=False),
    )
    if config.resume_training:
        wandb.init(project="pytorchlightning", resume=True)
        wandb.restore(config.ckpt_pth)
        models = ConditionalRelativeTransformer.load_from_checkpoint(config.ckpt_pth, config=config)
    else:
        models = model_factory.create(config.model_name, config)
    trainer.fit(models)


if __name__ == "__main__":
    known_args, _ = get_parser().parse_known_args()
    if known_args.model_type == ModelType.HuggingFace.value:
        main_hf(known_args)
    elif known_args.model_type == ModelType.PytorchLightning.value:
        main_pl(known_args)
