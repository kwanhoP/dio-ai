import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import tensorflow as tf
from transformers import EvaluationStrategy, GPT2Config
from transformers import Trainer as _Trainer
from transformers import TrainingArguments

from dioai.data.dataset import PozalabsDatasetFactory
from dioai.model import PozalabsModelFactory
from dioai.trainer import Trainer


@dataclass
class ModelConfig:
    min_length: int = 8
    n_vocab: int = 518
    n_ctx: int = 1024
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12


def get_parser() -> argparse.ArgumentParser:
    def bool_str(string: str) -> bool:
        string = str(string).lower()
        if string not in {"true", "false"}:
            raise ValueError("Not a valid boolean string.")
        return string == "true"

    parser = argparse.ArgumentParser("Train model")
    parser.add_argument("--data_dir", type=str, help="TFRecord 파일이 위치한 디렉토리")
    parser.add_argument("--output_dir", type=str, help="모델 체크포인트 저장 디렉토리")
    parser.add_argument("--model_config_path", type=str, help="모델 학습 config json 위치")
    parser.add_argument("--train_split", type=str, default="train", help="train dataset 식별자")
    parser.add_argument("--eval_split", type=str, default="dev", help="eval dataset 식별자")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--max_steps", type=int, default=50000, help="최대 학습 스텝")
    parser.add_argument("--eval_steps", type=int, default=1000, help="평가 수행 스텝 주기")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="GPU별 배치 사이즈")
    parser.add_argument("--logging_steps", type=int, default=100, help="학습 현황 로깅 스텝 주기")
    parser.add_argument("--model_name", type=str, default="gpt2_base", help="모델 이름")
    parser.add_argument(
        "--resume_training",
        type=bool_str,
        default=False,
        help="학습 재개 여부. `output_dir`에 이전 체크포인트가 있어야 합니다.",
    )
    parser.add_argument("--save_total_limit", type=int, default=10, help="저장할 최대 체크포인트 개수")
    parser.add_argument(
        "--logging_dir",
        type=str,
        help="텐서보드 로깅 디레토리. 기본값은 ${PWD}/runs/**CURRENT_DATETIME_HOSTNAME**",
    )
    return parser


def load_model_config(config_path) -> ModelConfig:
    with open(config_path, "r") as f_in:
        return ModelConfig(**json.load(f_in))


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


def main(args):
    # Tensorflow 의 GPU 메모리 독점을 막기 위한 GPU 메모리 점유 제한
    # https://github.com/tensorflow/tensorflow/issues/25138#issuecomment-583800729
    gpus = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_virtual_device_configuration(
        device=gpus[0],
        logical_devices=[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)],
    )

    data_dir = str(Path(args.data_dir).expanduser())
    model_config_path = str(Path(args.model_config_path).expanduser())
    output_dir = str(Path(args.output_dir).expanduser())

    model_config = load_model_config(model_config_path)

    model_name = args.model_name
    dataset_factory = PozalabsDatasetFactory()

    train_dataset = dataset_factory.create(
        name=model_name,
        data_dir=data_dir,
        split=args.train_split,
        min_length=model_config.min_length,
        max_length=model_config.n_ctx,
        shuffle=True,
    )
    eval_dataset = dataset_factory.create(
        name=model_name,
        data_dir=data_dir,
        split=args.eval_split,
        min_length=model_config.min_length,
        max_length=model_config.n_ctx,
        shuffle=False,
        training=False,
        bucket_by_sequence=False,
    )

    config = GPT2Config(
        vocab_size=model_config.n_vocab,
        n_ctx=model_config.n_ctx,
        n_embd=model_config.n_embd,
        n_head=model_config.n_head,
        n_layer=model_config.n_layer,
    )
    model_factory = PozalabsModelFactory()
    model = model_factory.create(model_name, config)

    training_args = TrainingArguments(
        output_dir,
        evaluation_strategy=EvaluationStrategy.STEPS,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        logging_dir=args.logging_dir,
    )

    trainer: _Trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset
    )
    trainer.train(
        resume_from_checkpoint=find_latest_checkpoint(output_dir) if args.resume_training else None
    )


if __name__ == "__main__":
    known_args, _ = get_parser().parse_known_args()
    main(known_args)
