import argparse
import json

import torch
from dataset import ChordProgressionSet
from model import Encoder
from torch import optim
from utils import encode_chord_progression


def train(args):
    backoffic_url = "https://backoffice.pozalabs.com/api/samples"
    # config
    with open(args.config, "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # midi 2 chord_token
    chord_token, n_vocab = encode_chord_progression(backoffic_url)

    # dataset
    dataset_train = ChordProgressionSet(chord_token)
    train_loader = torch.utils.data.DataLoader(dataset_train, args.batch_size, shuffle=True)

    # model
    config["n_enc_vocab"] = n_vocab + 1
    model = Encoder(config).to(device)

    # loss
    criterion = torch.nn.CrossEntropyLoss()

    # optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters)

    print(train_loader, criterion, optimizer, config, device)  # for test, 추후 삭제 예정
    for i, data in enumerate(train_loader):
        data = data.to(device)
        print(data.shape)  # for test, 추후 삭제 예정
        output, _ = model(data)
        print(output.shape)  # for test, 추후 삭제 예정


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="config.json", type=str, required=False, help="config file"
    )
    parser.add_argument("--batch_size", default=8, type=int, required=False, help="batch_size")
    parser.add_argument(
        "--weight_decay", type=float, default=0, required=False, help="weight decay"
    )

    args = parser.parse_args()

    train(args)
