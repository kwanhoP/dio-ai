from typing import Union

import numpy as np
import pytest

from dioai.preprocessor import chunk_midi


@pytest.mark.parametrize(
    "f, n, dtype, expected",
    [
        (0.1, 9, "float", 0.1),
        (0.1, 3, "str", "0.100"),
        (1e-2, 3, "float", 0.01),
        (1e-2, 3, "str", "0.010"),
    ],
)
def test_truncate(f: float, n: int, dtype: str, expected: Union[str, float]):
    assert chunk_midi.truncate(f, n, dtype) == expected


@pytest.mark.parametrize(
    "event_times, tempo_infos, end_time, expected",
    [
        (np.array([0.0]), np.array([120.0]), 60.0, 120),
        (np.array([0.0, 60.0]), np.array([100.0, 120.0]), 80.0, 105),
    ],
)
def test_get_avg_bpm(
    event_times: np.ndarray, tempo_infos: np.ndarray, end_time: float, expected: int
):
    result = chunk_midi.get_avg_bpm(
        event_times=event_times, tempo_infos=tempo_infos, end_time=end_time
    )
    assert result == expected
