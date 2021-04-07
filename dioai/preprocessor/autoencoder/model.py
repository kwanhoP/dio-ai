import layer
import torch
import torch.nn as nn
from torch import tensor


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.enc_emb = nn.Embedding(self.config["n_enc_vocab"], self.config["d_hidn"])
        positional_table = torch.FloatTensor(
            layer.get_positional_encoding_table(self.config["data_seq"] + 1, self.config["d_hidn"])
        )
        self.pos_emb = nn.Embedding.from_pretrained(positional_table, freeze=True)
        self.encoders = nn.ModuleList(
            [layer.EncoderLayer(self.config) for _ in range(self.config["n_layer"])]
        )

    def forward(self, inputs) -> tensor:
        positions = (
            torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype)
            .expand(inputs.size(0), inputs.size(1))
            .contiguous()
            + 1
        )
        pos_mask = inputs.eq(self.config["i_pad"])
        positions.masked_fill_(pos_mask, 0)
        outputs = self.enc_emb(inputs) + self.pos_emb(positions)
        attn_mask = layer.get_attn_pad_mask(inputs, inputs, self.config["i_pad"])

        attn_probs = []
        for encoder in self.encoders:
            outputs, attn_prob = encoder(outputs, attn_mask)
            attn_probs.append(attn_prob)
        return outputs, attn_probs
