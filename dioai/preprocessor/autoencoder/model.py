import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor

from dioai.preprocessor.autoencoder import layer
from dioai.preprocessor.autoencoder.dataset import ChordProgressionSet


class TransformerAutoEncoder(pl.LightningModule):
    def __init__(self, config, chord_token, batch_size):
        super().__init__()
        self.config = config
        self.chord_token = chord_token
        self.batch_size = batch_size
        self.enc_emb = nn.Embedding(
            self.config["n_enc_vocab"], self.config["d_hidn"], padding_idx=0
        )
        _positional_table = torch.FloatTensor(
            layer.get_positional_encoding_table(self.config["data_seq"] + 1, self.config["d_hidn"])
        )
        self.pos_emb = nn.Embedding.from_pretrained(_positional_table, freeze=True)
        self.encoder = layer.EncoderLayer(self.config)
        self.decoder = layer.DecoderLayer(self.config)
        self.learning_rate = config["learning_rate"]

    def forward(self, inputs) -> tensor:
        positions = (
            torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype)
            .expand(inputs.size(0), inputs.size(1))
            .contiguous()
            + 1
        )
        pos_mask = inputs.eq(self.config["i_pad"])
        positions.masked_fill_(pos_mask, 0)
        embedding_input = self.enc_emb(inputs) + self.pos_emb(positions)
        attn_mask = layer.get_attn_pad_mask(inputs, inputs, self.learning_rate)

        enc_output, _ = self.encoder(embedding_input, attn_mask)
        dec_output = self.decoder(enc_output)

        return dec_output, embedding_input

    def training_step(self, batch, idx):
        data = batch
        outputs, embedding_input = self(data)
        loss = F.mse_loss(outputs, embedding_input)
        tensor_board_logs = {"train_loss": loss, "lr": self.learning_rate}
        return {"loss": loss, "log": tensor_board_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        train_dataset = ChordProgressionSet(self.chord_token)
        train_loader = torch.utils.data.DataLoader(train_dataset, self.batch_size, shuffle=True)
        return train_loader


class GruAutoencoder(pl.LightningModule):
    def __init__(self, config, chord_token, batch_size):
        super().__init__()
        self.config = config
        self.chord_token = chord_token
        self.batch_size = batch_size
        self.learning_rate = config["learning_rate"]

        self.enc_emb = nn.Embedding(
            self.config["n_enc_vocab"], self.config["d_hidn"], padding_idx=0
        )
        self.encoder = nn.GRU(
            input_size=self.config["d_hidn"],
            hidden_size=self.config["d_hidn"],
            num_layers=1,
            dropout=config["dropout"],
        )
        self.decoder = layer.DecoderLayer(self.config)

    def forward(self, inputs) -> tensor:
        embedding_inputs = self.enc_emb(inputs)
        embedding_input = embedding_inputs.transpose(0, 1)

        _, output = self.encoder(embedding_input)
        output = self.decoder(output.view(-1, self.config["latent_dim"], self.config["d_hidn"]))
        return output, embedding_inputs

    def training_step(self, batch, idx):
        data = batch
        outputs, embedding_input = self(data)
        loss = F.mse_loss(outputs, embedding_input)
        tensor_board_logs = {"train_loss": loss, "lr": self.learning_rate}
        return {"loss": loss, "log": tensor_board_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        train_dataset = ChordProgressionSet(self.chord_token)
        train_loader = torch.utils.data.DataLoader(train_dataset, self.batch_size, shuffle=True)
        return train_loader

    def test_dataloader(self):
        test_dataset = ChordProgressionSet(self.chord_token)
        test_loader = torch.utils.data.DataLoader(test_dataset, 1, shuffle=False)
        return test_loader

    def encode_latent_vector(self, data):
        embedding_inputs = self.enc_emb(data)
        embedding_input = embedding_inputs.transpose(0, 1)
        _, output = self.encoder(embedding_input)
        return output
