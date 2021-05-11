import argparse
import datetime
import http
import json
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import requests
import torch

from dioai.config import TransformersConfig
from dioai.logger import logger
from dioai.model import PozalabsModelFactory
from dioai.preprocessor.encoder import BaseMetaEncoder, MetaEncoderFactory, decode_midi
from dioai.preprocessor.encoder.meta import (
    ATTR_ALIAS,
    META_ENCODING_ORDER,
    META_TO_ENCODER_ALIAS,
    Offset,
)
from dioai.preprocessor.utils import constants
from dioai.preprocessor.utils.container import MidiInfo, MidiMeta


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Midi generation")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="체크포인트 디렉토리")
    parser.add_argument("--output_dir", type=str, required=True, help="생성 결과물을 저장할 디렉토리")
    # Input meta
    parser.add_argument("--bpm", type=int)
    parser.add_argument("--audio_key", type=str, choices=list(constants.KEY_MAP.keys()))
    parser.add_argument("--time_signature", type=str, choices=list(constants.TIME_SIG_MAP.keys()))
    parser.add_argument("--pitch_range", type=str, choices=list(constants.PITCH_RANGE_MAP.keys()))
    parser.add_argument("--num_measures", type=int, choices=[4, 8])
    parser.add_argument(
        "--inst",
        type=str,
        choices=list(constants.PROGRAM_INST_MAP.keys()) + list(constants.POZA_INST_MAP.keys()),
    )
    parser.add_argument(
        "--genre", type=str, default="cinematic", choices=list(constants.GENRE_MAP.keys())
    )
    parser.add_argument("--min_velocity", type=int, default=40)
    parser.add_argument("--max_velocity", type=int, default=80)
    parser.add_argument(
        "--track_category", type=str, choices=list(constants.TRACK_CATEGORY_MAP.keys())
    )
    parser.add_argument(
        "--rhythm", type=str, default="standard", choices=list(constants.RHYTHM_MAP.keys())
    )
    parser.add_argument("--min_modulation", type=int, default=40)
    parser.add_argument("--max_modulation", type=int, default=80)
    parser.add_argument("--min_expression", type=int, default=40)
    parser.add_argument("--max_expression", type=int, default=80)
    parser.add_argument("--min_sustain", type=int, default=0)
    parser.add_argument("--max_sustain", type=int, default=127)
    # Sampling
    parser.add_argument("--num_generate", type=int, help="생성 개수")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--sample_id",
        type=str,
        help="생성에 사용할 코드 진행을 가져올 샘플 ID. 지정하지 않으면 학습에 사용된 코드 임베딩에서 무작위로 코드 진행을 선택합니다.",
    )
    return parser


def fetch_sample(sample_id) -> [Dict[str, Any]]:
    request_url = f"https://backoffice.pozalabs.com/api/samples/{sample_id}"
    response = requests.get(request_url)
    if response.status_code == http.HTTPStatus.OK:
        return response.json()["sample"]

    raise ValueError(f"Failed to fetch sample from backoffice. Status code: {response.status_code}")


def load_gen_config(checkpoint_dir: Union[str, Path]) -> Dict[str, Any]:
    with open(Path(checkpoint_dir).joinpath("config.json"), "r") as f_in:
        contents = json.load(f_in)
    return contents


def sub_offset(encoded_meta: torch.Tensor) -> Dict[str, int]:
    result = dict()
    for meta_name, value in zip(META_ENCODING_ORDER, encoded_meta.numpy().tolist()):
        if meta_name == "num_measures":
            # TODO: MidiInfo: `num_measure` 필드명 수정 (`MidiMeta`와 통일)
            # `MidiInfo`에서는 필드명이 `num_measure`
            result["num_measure"] = value
            continue

        meta_name_alias = ATTR_ALIAS.get(meta_name, meta_name)
        meta_name_alias = META_TO_ENCODER_ALIAS.get(meta_name_alias, meta_name_alias)
        offset = getattr(Offset, meta_name_alias.upper())
        adjusted_value = value - offset
        result[meta_name] = adjusted_value
    return result


def parse_meta(**kwargs: Any) -> MidiMeta:
    return MidiMeta(**kwargs)


def encode_meta(meta_encoder: BaseMetaEncoder, midi_meta: MidiMeta) -> List[int]:
    return meta_encoder.encode(midi_meta)


def generate_note_sequence(
    model,
    input_meta: torch.Tensor,
    num_generate: int,
    max_length: int,
    top_k: float,
    top_p: float,
    pad_token_id: int,
    eos_token_id: int,
    num_meta: int,
    chord_progression_vector: torch.Tensor,
):
    result = []
    for _ in range(num_generate):
        output = model.generate(
            input_meta,
            num_generate=1,
            do_sample=True,
            max_length=max_length,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            num_meta=torch.tensor([num_meta], dtype=torch.int64),
            chord_progression_vector=chord_progression_vector,
        )
        result.extend(output)
    return result


def decode_note_sequence(
    generation_result: List[torch.Tensor], num_meta: int, output_dir: Union[str, Path]
):
    for idx, raw_output in enumerate(generation_result):
        # TODO: 인덱싱으로 접근하지 않게 수정하기
        encoded_meta_dict = sub_offset(raw_output[:num_meta])
        note_sequence = raw_output[num_meta:]
        decode_midi(
            output_path=Path(output_dir).joinpath(f"decoded_{idx:03d}.mid"),
            midi_info=MidiInfo(**encoded_meta_dict, note_seq=note_sequence.numpy()),
        )


def load_chord_embedding(embedding_path: Union[str, Path]) -> Dict[Tuple, np.ndarray]:
    with open(embedding_path, "rb") as f_in:
        return pickle.load(f_in)


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    root_config_path = checkpoint_dir.parent.joinpath("root_config.json")
    config = TransformersConfig.from_file(root_config_path, from_pretrained=True)

    chord_embedding = None
    chord_progression = constants.UNKNOWN
    if config.chord_embedding_path is not None:
        chord_embedding = load_chord_embedding(config.chord_embedding_path)
        logger.info("`chord_embedding_path` is given. Chord progression will randomly selected")
        chord_progression = random.choice(list(chord_embedding.keys()))

    sample_id = args.sample_id
    if sample_id is not None:
        logger.info(f"Using chord progression from {sample_id}")
        try:
            sample = fetch_sample(sample_id)
            chord_progression = sample["chord_progressions"][0]
        except ValueError as exc:
            logger.info(str(exc))
            if config.chord_embedding_path is not None:
                logger.info("Chord progression will randomly selected")
                chord_progression = random.choice(list(chord_embedding.keys()))

    logger.info(f"Final chord progression: {chord_progression}")

    is_pozalabs_inst = args.inst in constants.POZA_INST_MAP
    dataset_name = "pozalabs" if is_pozalabs_inst else "reddit"
    logger.info(f"Using Encoder for {dataset_name}")

    meta_encoder_factory = MetaEncoderFactory()
    midi_meta = parse_meta(**vars(args), chord_progression=chord_progression)
    logger.info(f"Generating {args.num_generate} samples using following meta:\n{midi_meta.dict()}")

    encoded_meta = encode_meta(
        meta_encoder=meta_encoder_factory.create(dataset_name), midi_meta=midi_meta
    )
    logger.info("Encoded meta")

    logger.info("Start generation")
    model_factory = PozalabsModelFactory()
    generation_result = generate_note_sequence(
        model=model_factory.create(name=config.model_name, checkpoint_dir=checkpoint_dir),
        # 입력값은 [batch_size, sequence_length]
        input_meta=torch.unsqueeze(torch.LongTensor(encoded_meta), dim=0),
        num_generate=args.num_generate,
        top_k=args.top_k,
        top_p=args.top_p,
        max_length=config.model.n_ctx,
        pad_token_id=config.model.pad_token_id,
        eos_token_id=config.model.eos_token_id,
        num_meta=len(encoded_meta),
        chord_progression_vector=torch.tensor(
            chord_embedding[tuple(chord_progression)], dtype=torch.float32
        ),
    )
    logger.info("Finished generation")

    date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir.joinpath(date)
    output_dir.mkdir(exist_ok=True, parents=True)

    logger.info("Start decoding")
    decode_note_sequence(
        generation_result=generation_result,
        num_meta=len(encoded_meta),
        output_dir=output_dir,
    )
    logger.info("Finished decoding")


if __name__ == "__main__":
    known_args, _ = get_parser().parse_known_args()
    main(known_args)
