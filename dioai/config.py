from __future__ import annotations

import copy
import datetime
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Union

from transformers import EvaluationStrategy, GPT2Config, PretrainedConfig, TrainingArguments


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

    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> TransformersConfig:
        with open(json_path, "r") as f:
            data = json.load(f)

        start_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        output_root_dir = Path(data["output_root_dir"]).expanduser()
        output_root_dir.mkdir(exist_ok=True, parents=True)

        dir_name = f"{data['model_name']}-{start_time}"
        data = expanduser_data(data)
        model_config = GPT2Config(**data.pop("model"))
        training_config = TrainingArguments(
            **data.pop("training"),
            output_dir=str(output_root_dir.joinpath("checkpoints", dir_name)),
            logging_dir=str(output_root_dir.joinpath("checkpoints", dir_name)),
            evaluation_strategy=EvaluationStrategy.STEPS,
        )
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
            if key.endswith("_dir"):
                value = expanduser(value)
            copied_data[key] = value
    return copied_data


def expanduser(path: Union[str, Path]) -> str:
    return str(Path(path).expanduser())
