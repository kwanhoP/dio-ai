from torch.utils.data import DataLoader
from transformers import Trainer as _Trainer


class Trainer(_Trainer):
    def get_train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=None)
