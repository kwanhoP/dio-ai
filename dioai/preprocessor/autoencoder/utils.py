import http
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def encode_chord_progression(backoffice_url, pozalabs2_chord_progression_path) -> Tuple:
    poza_metas = load_poza_meta(backoffice_url)
    poza2 = np.load(
        pozalabs2_chord_progression_path,
        allow_pickle=True,
    )
    poza1 = []
    id_token = []
    for sample in poza_metas:
        chord_lst = np.array(sample["chord_progressions"]).flatten().tolist()
        if 0 < len(chord_lst) <= 128:
            poza1.append(np.array(sample["chord_progressions"]).flatten().tolist())
            id_token.append(sample["id"])

    tokenizer = Tokenizer()
    poza1_unique = list(set(map(tuple, poza1)))
    poza2_unique = list(set(map(tuple, poza2)))

    poza_total = poza1_unique + poza2_unique
    poza_total_unique = list(set(map(tuple, poza_total)))

    chord_progression_lst = [list(i) for i in poza_total_unique]
    tokenizer.fit_on_texts(chord_progression_lst)
    encoded_chord_token = tokenizer.texts_to_sequences(chord_progression_lst)
    padded_chord_token = pad_sequences(encoded_chord_token)

    return poza_total_unique, padded_chord_token, len(tokenizer.word_index)


def load_poza_meta(request_url: str, per_page: int = 1000) -> List[Dict[str, Any]]:
    return fetch_samples_from_backoffice(
        request_url=request_url, per_page=per_page, params={"auto_changed": False}
    )


def fetch_samples_from_backoffice(
    request_url: str, per_page: int = 100, params: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    page = 1
    result = []
    finished = False
    while not finished:
        has_next, samples = _fetch_samples(request_url, page=page, per_page=per_page, params=params)
        result.extend(samples)
        finished = not has_next
        page += 1
    return result


def _fetch_samples(
    url, page: int = 1, per_page: int = 100, params: Optional[Dict[str, Any]] = None
) -> Tuple[bool, List[Dict[str, Any]]]:
    request_params = {"page": page, "per_page": per_page}
    if params is not None:
        request_params.update(params)

    res = requests.get(url, params=request_params)

    if res.status_code != http.HTTPStatus.OK:
        raise ValueError("Failed to fetch samples from backoffice")

    data = res.json()
    return data["has_next"], data["samples"]["samples"]
