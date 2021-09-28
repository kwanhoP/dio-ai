import argparse
import datetime
import http
import pickle
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import requests
import torch

from dioai.config import TransformersConfig
from dioai.data.dataset.dataset import BartDenoisingNoteDataset
from dioai.logger import logger
from dioai.model import PozalabsModelFactory
from dioai.preprocessor.encoder import (
    BaseMetaEncoder,
    decode_midi,
    encode_midi,
    encode_remi_for_bart,
)
from dioai.preprocessor.utils import constants
from dioai.preprocessor.utils.container import MidiInfo, MidiMeta

MIDI_EXTENSIONS = (".mid", ".MID", ".midi", ".MIDI")
NUM_META = 19


@dataclass
class BriefMidiInfo(MidiInfo):
    """BART 는 메타 정보가 불필요하기 때문에 MidiInfo 를 None 으로 초기화합니다.
    마젠타 미디 디코딩에서 필요한 최소한의 정보 bpm, inst, time_signature, audio_key 만 받도록 합니다."""

    @classmethod
    def initialize(cls):
        args = [None] * NUM_META
        return cls(*args)

    def set_brief_infos(self, bpm: int, inst: int, time_signature: int, audio_key: int) -> Dict:
        self.bpm = bpm
        self.inst = inst
        self.time_signature = time_signature
        self.audio_key = audio_key
        return asdict(self)


def get_parser() -> argparse.ArgumentParser:
    # top_p: 0.5에서 1.5 사이 값으로 샘플마다 조절해서 생성해볼 필요가 있습니다.
    parser = argparse.ArgumentParser("Midi generation")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="체크포인트 디렉토리")
    parser.add_argument("--input_dir", type=str, required=True, help="디노이징 처리할 미디 파일 디렉토리")
    parser.add_argument("--output_dir", type=str, required=True, help="복원된 미디 파일을 저장할 디렉토리")
    parser.add_argument("--resolution", default=32, help="remi 전처리 시 인코딩/디코딩 resolution")
    parser.add_argument("--num_generate", type=int, default=5)
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
    parser.add_argument("--min_velocity", default=40)
    parser.add_argument("--max_velocity", default=80)
    parser.add_argument(
        "--track_category", type=str, choices=list(constants.TRACK_CATEGORY_MAP.keys())
    )
    parser.add_argument(
        "--rhythm", type=str, default="standard", choices=list(constants.RHYTHM_MAP.keys())
    )
    parser.add_argument("--min_modulation", default=constants.UNKNOWN)
    parser.add_argument("--max_modulation", default=constants.UNKNOWN)
    parser.add_argument("--min_expression", default=constants.UNKNOWN)
    parser.add_argument("--max_expression", default=constants.UNKNOWN)
    parser.add_argument("--min_sustain", default=constants.UNKNOWN)
    parser.add_argument("--max_sustain", default=constants.UNKNOWN)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument(
        "--sample_id",
        type=str,
        help="생성에 사용할 코드 진행을 가져올 샘플 ID. 지정하지 않으면 학습에 사용된 코드 임베딩에서 무작위로 코드 진행을 선택합니다.",
    )
    parser.add_argument(
        "--decoder_name", type=str, default="remi", help="decoder name ['remi','midi']"
    )
    return parser


def fetch_sample(sample_id) -> [Dict[str, Any]]:
    request_url = f"https://backoffice.pozalabs.com/api/samples/{sample_id}"
    response = requests.get(request_url)
    if response.status_code == http.HTTPStatus.OK:
        return response.json()["sample"]

    raise ValueError(f"Failed to fetch sample from backoffice. Status code: {response.status_code}")


def get_midi_paths(source_path: Union[str, Path]):
    midi_files = [
        filename for filename in source_path.rglob("**/*") if filename.suffix in MIDI_EXTENSIONS
    ]
    midi_files.sort()
    return midi_files


def confirm_special_tokens(note_seq: List[int], eos_id: int, bos_id: int):
    """샘플의 시작과 끝에 BOS 토큰과 EOS 토큰을 각각 추가합니다."""
    try:
        if note_seq[-1] != eos_id:
            note_seq = np.insert(note_seq, len(note_seq), eos_id)
        if note_seq[0] != bos_id:
            note_seq = np.insert(note_seq, 0, bos_id)
    except IndexError:
        return
    return note_seq


def remove_special_tokens(note_seq: List[int]):
    """마젠타에서 미디 디코딩 가능하도록 special tokens 가 있다면 제거합니다."""
    note_dictionary = BartDenoisingNoteDataset.get_note_dictionary()
    special_token_ids = (
        note_dictionary.bos_index,
        note_dictionary.eos_index,
        note_dictionary.pad_index,
        note_dictionary.mask_index,
    )
    note_seq = [int(vocab) for vocab in note_seq if vocab not in special_token_ids]
    note_seq.append(note_dictionary.eos_index)
    return note_seq


def parse_meta(**kwargs: Any) -> MidiMeta:
    return MidiMeta(**kwargs)


def encode_meta(meta_encoder: BaseMetaEncoder, midi_meta: MidiMeta) -> List[int]:
    return meta_encoder.encode(midi_meta)


def generate_note_seqs(
    model,
    corrupted_note_seq: torch.Tensor,
    num_generate: int,
    max_length: int,
    top_k: float,
    top_p: float,
    bos_token_id: int,
    pad_token_id: int,
    eos_token_id: int,
):
    result = []
    for _ in range(num_generate):
        output = model.generate(
            corrupted_note_seq,
            do_sample=True,
            max_length=max_length,
            top_k=top_k,
            top_p=top_p,
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        result.extend(output)
    return result


def decode_note_seqs(
    args, generation_results: List[torch.Tensor], output_dir: Union[str, Path], decoder_name: str
):
    brief_midi_info = BriefMidiInfo.initialize()
    brief_midi_info = brief_midi_info.set_brief_infos(
        bpm=args.bpm, inst=args.inst, time_signature=args.time_signature, audio_key=args.audio_key
    )
    for idx, raw_output in enumerate(generation_results):
        brief_midi_info["note_seq"] = list(raw_output)
        decode_midi(
            output_path=Path(output_dir).joinpath(f"denoised_{idx:03d}.mid"),
            midi_info=MidiInfo(**brief_midi_info),
            decoder_name=decoder_name,
        )


def load_chord_embedding(embedding_path: Union[str, Path]) -> Dict[Tuple, np.ndarray]:
    with open(embedding_path, "rb") as f_in:
        return pickle.load(f_in)


def main(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    root_config_path = checkpoint_dir.parent.joinpath("root_config.json")
    config = TransformersConfig.from_file(root_config_path, from_pretrained=True)

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

    midi_meta = parse_meta(**vars(args), chord_progression=chord_progression)
    sample_info = vars(midi_meta)
    sample_info["chord_progressions"] = sample_info.pop("chord_progression")

    midi_files = get_midi_paths(input_dir)
    encoded_midis = []

    for midi_file in midi_files:
        if args.decoder_name == "remi":
            note_seq = encode_remi_for_bart(
                midi_file, resolution=args.resolution, sample_info=sample_info
            )
        else:
            note_seq = encode_midi(midi_file, encoder_name=args.decoder_name)

        note_seq = confirm_special_tokens(
            note_seq, bos_id=config.model.bos_token_id, eos_id=config.model.eos_token_id
        )
        encoded_midis.append(note_seq)
    logger.info("Start generation")

    # 디노이징을 진행할 타겟 미디 파일
    model_factory = PozalabsModelFactory()
    results = []
    for encoded_midi in encoded_midis:
        tmp_results = generate_note_seqs(
            model=model_factory.create(name=config.model_name, checkpoint_dir=checkpoint_dir),
            corrupted_note_seq=torch.unsqueeze(torch.LongTensor(encoded_midi), dim=0),
            num_generate=args.num_generate,
            top_k=args.top_k,
            top_p=args.top_p,
            max_length=config.model.max_position_embeddings,
            bos_token_id=config.model.bos_token_id,
            pad_token_id=config.model.pad_token_id,
            eos_token_id=config.model.eos_token_id,
        )
        results.append(tmp_results[0])

    logger.info("Finished generation")

    logger.info("Start decoding")
    generation_results = []
    for denoised_note_seq in results:
        # 마젠타에서 디코딩 가능한 토큰으로 구성합니다.
        denoised_note_seq = remove_special_tokens(denoised_note_seq)
        generation_results.append(denoised_note_seq)

    date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir.joinpath(date)
    output_dir.mkdir(exist_ok=True, parents=True)

    decode_note_seqs(
        args,
        generation_results=generation_results,
        output_dir=output_dir,
        decoder_name=args.decoder_name,
    )
    logger.info("Finished decoding")


if __name__ == "__main__":
    known_args, _ = get_parser().parse_known_args()
    main(known_args)
