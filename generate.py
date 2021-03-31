import argparse
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Union

import torch

from dioai.logger import logger
from dioai.model.model import GP2MetaToNoteModel
from dioai.preprocessor.encoder import BaseMetaEncoder, MetaEncoderFactory, decode_midi
from dioai.preprocessor.encoder.meta import META_ENCODING_ORDER, Offset
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
    parser.add_argument("--inst", type=str, choices=list(constants.PROGRAM_INST_MAP.keys()))
    # Sampling
    parser.add_argument("--num_generate", type=int, help="생성 개수")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    return parser


def load_gen_config(checkpoint_dir: Union[str, Path]) -> Dict[str, Any]:
    with open(Path(checkpoint_dir).joinpath("config.json"), "r") as f_in:
        contents = json.load(f_in)
    return contents


def sub_offset(encoded_meta: torch.Tensor) -> List[int]:
    result = []
    for meta_name, value in zip(META_ENCODING_ORDER, encoded_meta.numpy().tolist()):
        if meta_name == "num_measures":
            result.append(value)
            continue

        offset = getattr(Offset, meta_name.upper())
        adjusted_value = value - offset
        result.append(adjusted_value)
    return result


def encode_meta(
    meta_encoder: BaseMetaEncoder,
    bpm: int,
    audio_key: str,
    time_signature: str,
    pitch_range: str,
    num_measures: int,
    inst: str,
) -> List[int]:
    midi_meta = MidiMeta(
        bpm=bpm,
        audio_key=audio_key,
        time_signature=time_signature,
        pitch_range=pitch_range,
        num_measures=num_measures,
        inst=inst,
    )
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
        )
        result.extend(output)
    return result


def decode_note_sequence(
    generation_result: List[torch.Tensor], num_meta: int, output_dir: Union[str, Path]
):
    for idx, raw_output in enumerate(generation_result):
        # TODO: 인덱싱으로 접근하지 않게 수정하기
        bpm, audio_key, time_signature, pitch_range, num_measures, inst = sub_offset(
            raw_output[:num_meta]
        )
        note_sequence = raw_output[num_meta:] - constants.META_LEN
        decode_midi(
            output_path=Path(output_dir).joinpath(f"decoded_{idx:03d}.mid"),
            midi_info=MidiInfo(
                bpm=bpm,
                audio_key=audio_key,
                time_signature=time_signature,
                pitch_range=pitch_range,
                num_measure=num_measures,  # 디코딩시 사용되지 않는 값
                inst=inst,
                note_seq=note_sequence.numpy(),
            ),
        )


def main(args: argparse.Namespace) -> None:
    checkpoint_dir = Path(known_args.checkpoint_dir).expanduser()
    output_dir = Path(known_args.output_dir).expanduser()

    encoded_meta = encode_meta(
        meta_encoder=MetaEncoderFactory().create("reddit"),
        bpm=args.bpm,
        audio_key=args.audio_key,
        time_signature=args.time_signature,
        pitch_range=args.pitch_range,
        num_measures=args.num_measures,
        inst=args.inst,
    )
    logger.info("Encoded meta")

    logger.info("Start generation")
    gen_config = load_gen_config(checkpoint_dir)
    generation_result = generate_note_sequence(
        model=GP2MetaToNoteModel.from_pretrained(checkpoint_dir),
        # 입력값은 [batch_size, sequence_length]
        input_meta=torch.unsqueeze(torch.LongTensor(encoded_meta), dim=0),
        num_generate=args.num_generate,
        top_k=args.top_k,
        top_p=args.top_p,
        max_length=gen_config["n_ctx"],
        pad_token_id=gen_config["pad_token_id"],
        eos_token_id=gen_config["eos_token_id"],
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
