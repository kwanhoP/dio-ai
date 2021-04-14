import inspect
from typing import Any, Dict


def inject_args(func, **kwargs) -> Dict[str, Any]:
    args = [arg for arg in inspect.getfullargspec(func).args]
    return {key: value for key, value in kwargs.items() if key in args}
