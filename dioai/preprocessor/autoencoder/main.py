import argparse
import json
import pickle

import numpy as np
from pytorch_lightning import Trainer

from dioai.preprocessor.autoencoder import model, utils


def train(args, config, chord_token):
    trainer = Trainer(
        gpus=1,
        max_epochs=100,
        fast_dev_run=False,
        default_root_dir="/media/experiments/checkpoints/autoencoder_chord_to_vector",
    )
    if args.model == "gru":
        models = model.GruAutoencoder(config, chord_token, args.batch_size)
    elif args.model == "transformer":
        models = model.TransformerAutoEncoder(config, chord_token, args.batch_size)
    trainer.fit(models)


def inference(args, config, chord_token, raw_chord_progression):
    if args.model == "gru":
        models = model.GruAutoencoder.load_from_checkpoint(
            args.ckpt_path, config=config, chord_token=chord_token, batch_size=1
        )
        models.eval()
        data_loader = models.test_dataloader()
        chord_embedding_vector = np.zeros(shape=(len(chord_token), config["d_hidn"]))
        for idx, data in enumerate(data_loader):
            chord_embedding_vector[idx] = models.encode_latent_vector(data).detach().numpy()

        chord_embedding_table = dict(zip(raw_chord_progression, chord_embedding_vector))

        with open(
            "/media/data/dioai/pozalabs/chord_embedding_table/chord_embedding_table.pickle", "wb"
        ) as fw:
            pickle.dump(chord_embedding_table, fw)

    elif args.model == "transformer":
        models = model.TransformerAutoEncoder(args.ckpt_path)


def main(args):
    BACKOFFICE_URL = "https://backoffice.pozalabs.com/api/samples"
    with open(args.config, "r") as f:
        config = json.load(f)
    raw_chord_progression, chord_token, n_vocab = utils.encode_chord_progression(
        BACKOFFICE_URL, args.train
    )
    config["n_enc_vocab"] = n_vocab + 1

    if args.train:
        train(args, config, chord_token)
    else:
        inference(args, config, chord_token, raw_chord_progression)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="autoencoder_config.json", type=str, required=False, help="config file"
    )
    parser.add_argument("--batch_size", default=8, type=int, required=False, help="batch_size")
    parser.add_argument(
        "--model",
        default="gru",
        choices=["gru", "transformer"],
        type=str,
        required=False,
        help="오토인코더 모델 선택",
    )
    parser.add_argument(
        "--ckpt_path",
        default="",
        type=str,
        required=False,
        help="ckpt 경로",
    )
    parser.add_argument(
        "--train",
        default=False,
        type=bool,
        required=False,
        help="True: 학습모드, False: inference 모드",
    )

    args = parser.parse_args()
    main(args)
