import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf
from transformers import EvaluationStrategy, GPT2Config, GPT2LMHeadModel
from transformers import Trainer as _Trainer
from transformers import TrainingArguments

from dioai.data.dataset.gpt2 import GPT2Dataset
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
    parser.add_argument(
        "--resume_training",
        type=bool_str,
        default=False,
        help="학습 재개 여부. `output_dir`에 이전 체크포인트가 있어야 합니다.",
    )
    return parser


def load_model_config(config_path) -> ModelConfig:
    with open(config_path, "r") as f_in:
        return ModelConfig(**json.load(f_in))


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

    train_dataset = GPT2Dataset(
        data_dir=data_dir,
        split=args.train_split,
        min_length=model_config.min_length,
        max_length=model_config.n_ctx,
        shuffle=True,
    )
    eval_dataset = GPT2Dataset(
        data_dir=data_dir,
        split=args.eval_split,
        min_length=model_config.min_length,
        max_length=model_config.n_ctx,
        shuffle=False,
        training=False,
        bucket_by_sequence=False,
    ).to_dataset()

    config = GPT2Config(
        vocab_size=model_config.n_vocab,
        n_ctx=model_config.n_ctx,
        n_embd=model_config.n_embd,
        n_head=model_config.n_head,
        n_layer=model_config.n_layer,
    )
    model = GPT2LMHeadModel(config)

    training_args = TrainingArguments(
        output_dir,
        evaluation_strategy=EvaluationStrategy.STEPS,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        logging_steps=args.logging_steps,
    )

    trainer: _Trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset
    )
    trainer.train(resume_from_checkpoint=args.resume_training if args.resume_training else None)


if __name__ == "__main__":
    known_args, _ = get_parser().parse_known_args()
    main(known_args)
