from __future__ import annotations

import copy
import datetime
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

from transformers import GPT2Config, PretrainedConfig, TrainingArguments


@dataclass
class TransformersConfig:
    model_name: str
    data_dir: str
    train_split: str
    eval_split: str
    test_split: str
    resume_training: str
    tf_gpu_memory_limit: int
    model: PretrainedConfig
    training: TrainingArguments
    output_root_dir: str
    logging_root_dir: str
    num_meta: Optional[int] = None
    chord_embedding_path: Optional[str] = None

    @classmethod
    def from_file(
        cls, json_path: Union[str, Path], from_pretrained: bool = False
    ) -> TransformersConfig:
        with open(json_path, "r") as f:
            data = json.load(f)

        output_root_dir = Path(data["output_root_dir"]).expanduser()
        logging_root_dir = Path(data["logging_root_dir"]).expanduser()
        resume_training = data["resume_training"]

        if resume_training or from_pretrained:
            output_dir = output_root_dir
            logging_dir = logging_root_dir
        else:
            start_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
            dir_name = f"{data['model_name']}-{start_time}"
            output_dir = output_root_dir.joinpath("checkpoints", dir_name)
            output_dir.mkdir(exist_ok=True, parents=True)
            logging_dir = logging_root_dir.joinpath("runs", dir_name)
            logging_dir.mkdir(exist_ok=True, parents=True)

        data = expanduser_data(data)
        model_config = GPT2Config(**data.pop("model"))

        training_arguments_dict = data.pop("training")
        if training_arguments_dict.get("output_dir") is None:
            training_arguments_dict.update(
                {"output_dir": str(output_dir), "logging_dir": str(logging_dir)}
            )
        valid_training_arguments_dict = {
            key: value for key, value in training_arguments_dict.items() if not key.startswith("_")
        }
        training_config = TrainingArguments(**valid_training_arguments_dict)
        return cls(**data, model=model_config, training=training_config)

    def save(self) -> None:
        output_path = Path(self.training.output_dir).joinpath("root_config.json")
        output_path.parent.mkdir(exist_ok=True, parents=True)
        with open(output_path, "w") as f:
            json.dump(self.dict(), f, indent=2)

    def dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["training"]["evaluation_strategy"] = self.training.evaluation_strategy.value
        data["training"]["lr_scheduler_type"] = self.training.lr_scheduler_type.value
        data["model"] = json.loads(data["model"].to_json_string())
        return data

    @property
    def batch_size(self) -> int:
        try:
            import tensorflow as tf

            num_gpu = len(tf.config.list_physical_devices("GPU"))
        except ImportError:
            import torch

            num_gpu = torch.cuda.device_count()
        return int(self.training.per_device_train_batch_size * num_gpu)


def expanduser_data(data: Dict[str, Any]) -> Dict[str, Any]:
    copied_data = copy.deepcopy(data)
    for key, value in data.items():
        if isinstance(value, dict):
            copied_data[key] = expanduser_data(value)
        else:
            if key.endswith("_dir") and isinstance(value, str):
                value = expanduser(value)
            copied_data[key] = value
    return copied_data


def expanduser(path: Union[str, Path]) -> str:
    return str(Path(path).expanduser())
