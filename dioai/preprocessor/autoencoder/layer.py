import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor


def get_positional_encoding_table(n_seq: int, d_hidn: int) -> tensor:
    def cal_angle(position, i_hidn):
        return position / np.power(10000, 2 * (i_hidn // 2) / d_hidn)

    def get_posi_angle_vec(position):
        return [cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]

    positional_table = np.array([get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
    positional_table[:, 0::2] = np.sin(positional_table[:, 0::2])
    positional_table[:, 1::2] = np.cos(positional_table[:, 1::2])

    return positional_table


def get_attn_pad_mask(seq_q: int, seq_k: int, i_pad: int) -> tensor:
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(i_pad).unsqueeze(1).expand(batch_size, len_q, len_k)  # <pad>
    return pad_attn_mask


class ScaledAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config["dropout"])
        self.scale = 1 / (self.config["d_head"] ** 0.5)

    def forward(self, Q: tensor, K: tensor, V: tensor, attn_mask: tensor) -> tensor:
        scores = torch.matmul(Q, K.transpose(-1, -2)).mul_(self.scale)
        scores.masked_fill_(attn_mask, -1e9)
        attn_prob = nn.Softmax(dim=-1)(scores)
        attn_prob = self.dropout(attn_prob)
        context = torch.matmul(attn_prob, V)
        return context, attn_prob


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.W_Q = nn.Linear(self.config["d_hidn"], self.config["n_head"] * self.config["d_head"])
        self.W_K = nn.Linear(self.config["d_hidn"], self.config["n_head"] * self.config["d_head"])
        self.W_V = nn.Linear(self.config["d_hidn"], self.config["n_head"] * self.config["d_head"])
        self.scaled_attn = ScaledAttentionLayer(self.config)
        self.linear = nn.Linear(
            self.config["n_head"] * self.config["d_head"], self.config["d_hidn"]
        )
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, Q: tensor, K: tensor, V: tensor, attn_mask: tensor) -> tensor:
        batch_size = Q.size(0)

        q_s = (
            self.W_Q(Q)
            .view(batch_size, -1, self.config["n_head"], self.config["d_head"])
            .transpose(1, 2)
        )
        k_s = (
            self.W_K(K)
            .view(batch_size, -1, self.config["n_head"], self.config["d_head"])
            .transpose(1, 2)
        )
        v_s = (
            self.W_V(V)
            .view(batch_size, -1, self.config["n_head"], self.config["d_head"])
            .transpose(1, 2)
        )
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.config["n_head"], 1, 1)
        context, attn_prob = self.scaled_attn(q_s, k_s, v_s, attn_mask)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.config["n_head"] * self.config["d_head"])
        )

        output = self.linear(context)
        output = self.dropout(output)

        return output, attn_prob


class PoswiseFeedForwardLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.fc1 = nn.Conv1d(
            in_channels=self.config["data_seq"],
            out_channels=self.config["data_seq"] * 4,
            kernel_size=1,
        )
        self.fc2 = nn.Conv1d(
            in_channels=self.config["data_seq"] * 4,
            out_channels=self.config["data_seq"] // 2,
            kernel_size=1,
        )
        self.fc3 = nn.Conv1d(
            in_channels=self.config["data_seq"] // 2,
            out_channels=self.config["latent_dim"],
            kernel_size=1,
        )
        self.active = F.gelu
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, inputs: tensor) -> tensor:

        output = self.active(self.fc1(inputs))
        output = self.active(self.fc2(output))
        output = self.fc3(output)
        output = self.dropout(output)

        return output


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.self_attn = MultiHeadAttentionLayer(self.config)
        self.layer_norm1 = nn.LayerNorm(
            self.config["d_hidn"], eps=self.config["layer_norm_epsilon"]
        )
        self.fc = PoswiseFeedForwardLayer(self.config)

    def forward(self, inputs: tensor, attn_mask: tensor) -> tensor:
        att_outputs, attn_prob = self.self_attn(inputs, inputs, inputs, attn_mask)
        att_outputs = self.layer_norm1(inputs + att_outputs)
        fc_outputs = self.fc(att_outputs)
        return fc_outputs, attn_prob
