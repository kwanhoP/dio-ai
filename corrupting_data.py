import argparse
import datetime
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Union

import torch
import transformers

import train
from dioai.data.dataset import PozalabsDatasetFactory
from dioai.data.dataset.dataset import BartDenoisingNoteDataset
from dioai.logger import logger
from dioai.model import PozalabsModelFactory
from dioai.preprocessor.encoder import decode_midi
from dioai.preprocessor.utils import constants
from dioai.preprocessor.utils.container import MidiInfo
from dioai.trainer import Trainer_hf

MIDI_EXTENSIONS = (".mid", ".MID", ".midi", ".MIDI")
NUM_META = 19
BPM = 150 // 5
AUDIO_KEY = 0
INST = 0
TIME_SIGNATURE = 4


class BriefMidiInfo(MidiInfo):
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
    parser = argparse.ArgumentParser("Generate corrupt data from BART")
    parser.add_argument("--output_dir", type=str, help="corrupted된 data를 저장할 위치")
    parser.add_argument("--config_path", type=str, help="전체 설정값이 저장된 JSON 파일 경로")
    parser.add_argument("--bpm", type=int, default=BPM)
    parser.add_argument(
        "--audio_key", type=str, default=AUDIO_KEY, choices=list(constants.KEY_MAP.keys())
    )
    parser.add_argument(
        "--time_signature",
        type=str,
        default=TIME_SIGNATURE,
        choices=list(constants.TIME_SIG_MAP.keys()),
    )
    parser.add_argument(
        "--inst",
        type=str,
        default=INST,
        choices=list(constants.PROGRAM_INST_MAP.keys()) + list(constants.POZA_INST_MAP.keys()),
    )
    parser.add_argument(
        "--decoder_name", type=str, default="remi", help="decoder name ['remi','midi']"
    )
    return parser


def remove_special_tokens(note_seq: List[int]):
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
            output_path=Path(output_dir).joinpath(f"corrupted_{idx:03d}.mgit gasdgasdgagid"),
            midi_info=MidiInfo(**brief_midi_info),
            decoder_name=decoder_name,
        )


def main(args):
    output_dir = Path(args.output_dir).expanduser()
    config = train.load_config(Path(args.config_path).expanduser(), model_type="hf")

    dataset_factory = PozalabsDatasetFactory()
    model_factory = PozalabsModelFactory()

    dataset = dataset_factory.create(
        config=config,
        split=config.train_split,
    )
    dataset.tf_dataset_build_args["training"] = False
    trainer: transformers.Trainer = Trainer_hf(
        model=model_factory.create(config.model_name, config.model),
        args=config.training,
        train_dataset=dataset,
    )

    data = trainer.get_train_dataloader()

    entire_data = []
    logger.info("Start corrupting")
    for i in data:
        tmp_data = next(iter(data))["input_ids"]
        entire_data.append(tmp_data)
    logger.info("Finish corrupting")

    generation_results = []
    logger.info("Start decoding")
    for denoised_note_seq in entire_data:
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
    logger.info("Finish decoding")


if __name__ == "__main__":
    known_args, _ = get_parser().parse_known_args()
    main(known_args)
