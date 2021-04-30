from typing import Optional

from torch.utils.data.dataloader import DataLoader
from transformers import Trainer as _Trainer
from transformers.integrations import is_fairscale_available
from transformers.optimization import (
    Adafactor,
    AdamW,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_scheduler,
)

if is_fairscale_available():
    from fairscale.optim import OSS


class Trainer(_Trainer):
    def __init__(
        self,
        use_cosine_annealing: Optional[bool] = None,
        num_cycles: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_cosine_annealing = use_cosine_annealing
        self.num_cycles = num_cycles

    def get_train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=None)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        # default optimizer
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            optimizer_cls = Adafactor if self.args.adafactor else AdamW
            if self.args.adafactor:
                optimizer_cls = Adafactor
                optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
            else:
                optimizer_cls = AdamW
                optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                }
            optimizer_kwargs["lr"] = self.args.learning_rate
            if self.sharded_dpp:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        # cosine anneling lr_scheduler 적용을 위해 overiding
        if self.use_cosine_annealing:
            self.lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=self.num_cycles,
            )
        else:
            self.lr_scheduler = get_scheduler(
                name=self.args.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
            )
