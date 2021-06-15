import argparse
import re
from functools import partial
from pathlib import Path
from typing import Dict, Union

import faiss
import numpy as np
import pandas as pd
import torch
from datasets import Features, Sequence, Value, load_dataset
from keras_preprocessing.sequence import pad_sequences

from dioai.config import PytorchlightConfig, TransformersConfig
from dioai.model import ModelType, PozalabsModelFactory


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("save dpr dataset & faiss index")
    parser.add_argument("--config_path", type=str, help="전체 설정값이 저장된 JSON 파일 경로")
    parser.add_argument("--embed_dim", default=768, type=int, help="모델 임배딩 차원")
    parser.add_argument("--csv_path", type=str, help="저장할 데이터 셋 csv 변환 파일 경로")
    parser.add_argument("--dataset_out", default="note_dpr_dataset", type=str, help="dataset 저장 경로")
    parser.add_argument("--index_out", default="note_dpr_index.faiss", type=str, help="index 저장 경로")
    parser.add_argument("--gpu_num", default=0, type=int, help="사용할 gpu 번호", choices=[0, 1, 2, 3])
    return parser


# config utils
def find_latest_checkpoint(checkpoint_dir: Union[str, Path]) -> str:
    def _sort_by_checkpoint(_p: Path) -> int:
        return int(_p.stem.split("-")[-1])

    # 체크포인트는 `checkpoint-{step}`으로 저장됨
    sorted_checkpoints = sorted(
        (
            dir_name
            for dir_name in Path(checkpoint_dir).iterdir()
            if dir_name.is_dir() and dir_name.stem.startswith("checkpoint")
        ),
        key=_sort_by_checkpoint,
    )
    return str(sorted_checkpoints[-1])


def load_config(config_path: Union[str, Path], model_type: str) -> TransformersConfig:
    if model_type == ModelType.HuggingFace.value:
        config = TransformersConfig.from_file(config_path)
        config.fine_tune_ckpt = (
            find_latest_checkpoint(config.output_root_dir) if config.resume_training else None
        )
        config.save()
    elif model_type == ModelType.PytorchLightning.value:
        config = PytorchlightConfig.from_file(config_path)
    return config


# data
def mk_dpr_csv(note_data_path: str) -> None:
    """
    dpr dataset 형식은 text를 columns으로 하는 csv 파일이 필요
    np.ndarray 형식의 note를 csv로 저장하는 함수
    """
    raw_note = np.load(note_data_path, allow_pickle=True)
    new_note = []
    for note in raw_note:
        new_note.append(np.insert(note, 0, 2))
    new_note = np.array(new_note)
    new_note = pad_sequences(new_note, maxlen=512, padding="post")
    note = torch.unsqueeze(torch.tensor(new_note), 1)
    pd.DataFrame(note, columns=["text"]).to_csv(
        "note_dpr.csv", index=False, index_label=False, header=False
    )


# embed
def embed(documents: dict, encoder, device: str) -> Dict[str : np.ndarray]:
    """
    csv로 저장한 note_seq를 불러와 tensor로 변환 후 pretrained model에 임배딩 시킨다
    """
    raw_data = documents["text"]
    input_ids = (
        torch.tensor(list(map(int, re.findall(r"\d+", str(raw_data)))))
        .long()[:-1]
        .view(1, -1)
        .to(device)
    )
    encoder = encoder.to(device)
    embeddings = encoder(input_ids, return_dict=True).pooler_output
    embeddings = embeddings.squeeze()
    return {"embeddings": embeddings.cpu().numpy().astype("float32")}


def save_note_dpr_index(args):
    config_path = args.config_path
    embed_dim = args.embed_dim
    csv_path = args.csv_path
    dataset_out = args.dataset_out
    index_out = args.index_out
    gpu_num = args.gpu_num

    # set GPU
    torch.set_grad_enabled(False)
    device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    # load pretrained dpr_ctx_encoder
    model_factory = PozalabsModelFactory()
    config = load_config(Path(config_path).expanduser(), "hf")
    dpr_model = model_factory.create(config.model_name, config.model)
    dpr_pretrained = dpr_model.from_pretrained(
        "/media/experiments/checkpoints/dpr_model_hf-210614_134529/checkpoint-15000"
    )

    # load dataset
    dataset = load_dataset("csv", data_files=[csv_path], split="train", column_names=["text"])

    new_features = Features({"text": Value("string"), "embeddings": Sequence(Value("float32"))})

    # embed note using pretriained dpr_note_encoder
    dataset = dataset.map(
        partial(embed, encoder=dpr_pretrained.dpr_note_encoder, device=device),
        features=new_features,
    )

    # save embedding dataset
    dataset.save_to_disk(dataset_out)

    # indexing dataset & save faiss index
    index = faiss.IndexHNSWFlat(embed_dim, 128, faiss.METRIC_INNER_PRODUCT)
    dataset.add_faiss_index("embeddings", custom_index=index)
    dataset.get_index("embeddings").save(index_out)


if __name__ == "__main__":
    known_args, _ = get_parser().parse_known_args()
    save_note_dpr_index(known_args)
