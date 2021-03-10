import inspect
import pprint

from .dataset import GPT2BaseDataset


class PozalabsDatasetFactory:
    dataset_map = {GPT2BaseDataset.name: GPT2BaseDataset}

    def create(self, name: str, *args, **kwargs):
        dataset_cls = self.dataset_map.get(name)
        if dataset_cls is None:
            raise ValueError(f"`name` should be one of {tuple(self.dataset_map.keys())}")
        try:
            instance = dataset_cls(*args, **kwargs)
        except TypeError as exc:
            full_arg_spec = inspect.getfullargspec(dataset_cls.__init__)
            raise TypeError(
                f"Failed to initialize {dataset_cls.__class__.__name__} due to invalid arguments. "
                f"Check all required arguments are passed. Required args:\n"
                f"{pprint.pformat(full_arg_spec.annotations, indent=2)}"
            ) from exc
        return instance.build()


__all__ = ["GPT2BaseDataset", "PozalabsDatasetFactory"]
