import copy
import math
import math as m
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


def get_masked_with_pad_tensor(size, src, trg, pad_token):
    """
    :param size: the size of target input
    :param src: source tensor
    :param trg: target tensor
    :param pad_token: pad token
    """
    src = src[:, None, None, :]
    trg = trg[:, None, None, :]
    src_pad_tensor = torch.ones_like(src).to(src.device.type) * pad_token
    src_mask = torch.equal(src, src_pad_tensor)
    trg_mask = torch.equal(src, src_pad_tensor)
    if trg is not None:
        trg_pad_tensor = torch.ones_like(trg).to(trg.device.type) * pad_token
        dec_trg_mask = trg == trg_pad_tensor
        # boolean reversing i.e) True * -1 + 1 = False
        seq_mask = ~sequence_mask(torch.arange(1, size + 1).to(trg.device), size)
        # look_ahead_mask = torch.max(dec_trg_mask, seq_mask)
        look_ahead_mask = dec_trg_mask | seq_mask

    else:
        trg_mask = None
        look_ahead_mask = None

    return src_mask, trg_mask, look_ahead_mask


def sequence_mask(length, max_length=None):
    """Tensorflow의 sequence_mask를 구현"""
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


class DynamicPositionEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim, max_seq=512):
        super().__init__()
        embed_sinusoid_list = np.array(
            [
                [
                    [
                        m.sin(
                            pos
                            * m.exp(-m.log(10000) * i / embedding_dim)
                            * m.exp(m.log(10000) / embedding_dim * (i % 2))
                            + 0.5 * m.pi * (i % 2)
                        )
                        for i in range(embedding_dim)
                    ]
                    for pos in range(max_seq)
                ]
            ]
        )
        self.positional_embedding = embed_sinusoid_list

    def forward(self, x):
        x = x + torch.from_numpy(self.positional_embedding[:, : x.size(1), :]).to(
            x.device, dtype=x.dtype
        )
        return x


class RelativeGlobalAttention(torch.nn.Module):
    """
    ref) Music Transformer ( Huang et al, 2018 )
    [paper link](https://arxiv.org/pdf/1809.04281.pdf)
    """

    def __init__(self, h=4, d=256, add_emb=False, max_seq=512, **kwargs):
        super().__init__()
        self.len_k = None
        self.max_seq = max_seq
        self.E = None
        self.h = h
        self.d = d
        self.dh = d // h
        self.Wq = torch.nn.Linear(self.d, self.d)
        self.Wk = torch.nn.Linear(self.d, self.d)
        self.Wv = torch.nn.Linear(self.d, self.d)
        self.fc = torch.nn.Linear(d, d)
        self.additional = add_emb
        self.E = torch.randn([self.max_seq, int(self.dh)], requires_grad=False)
        if self.additional:
            self.Radd = None

    def forward(self, inputs, mask=None, **kwargs):
        """
        :param inputs: a list of tensors. i.e) [Q, K, V]
        :param mask: mask tensor
        :param kwargs:
        :return: final tensor ( output of attention )
        """
        q = inputs[0]
        q = self.Wq(q)
        q = torch.reshape(q, (q.size(0), q.size(1), self.h, -1))
        q = q.permute(0, 2, 1, 3)  # batch, h, seq, dh

        k = inputs[1]
        k = self.Wk(k)
        k = torch.reshape(k, (k.size(0), k.size(1), self.h, -1))
        k = k.permute(0, 2, 1, 3)

        v = inputs[2]
        v = self.Wv(v)
        v = torch.reshape(v, (v.size(0), v.size(1), self.h, -1))
        v = v.permute(0, 2, 1, 3)

        self.len_k = k.size(2)
        self.len_q = q.size(2)

        E = self._get_left_embedding(self.len_q, self.len_k).to(q.device)
        QE = torch.einsum("bhld,md->bhlm", [q, E])
        QE = self._qe_masking(QE)
        Srel = self._skewing(QE)

        Kt = k.permute(0, 1, 3, 2)
        QKt = torch.matmul(q, Kt)
        logits = QKt + Srel
        logits = logits / math.sqrt(self.dh)

        if mask is not None:
            logits += (mask.to(torch.int64) * -1e9).to(logits.dtype)

        attention_weights = F.softmax(logits, -1)
        attention = torch.matmul(attention_weights, v)

        out = attention.permute(0, 2, 1, 3)
        out = torch.reshape(out, (out.size(0), -1, self.d))

        out = self.fc(out)
        return out, attention_weights

    def _get_left_embedding(self, len_q, len_k):
        starting_point = max(0, self.max_seq - len_q)
        e = self.E[starting_point:, :]
        return e

    def _skewing(self, tensor: torch.Tensor):
        padded = F.pad(tensor, [1, 0, 0, 0, 0, 0, 0, 0])
        reshaped = torch.reshape(
            padded, shape=[padded.size(0), padded.size(1), padded.size(-1), padded.size(-2)]
        )
        Srel = reshaped[:, :, 1:, :]
        if self.len_k > self.len_q:
            Srel = F.pad(Srel, [0, 0, 0, 0, 0, 0, 0, self.len_k - self.len_q])
        elif self.len_k < self.len_q:
            Srel = Srel[:, :, :, : self.len_k]

        return Srel

    @staticmethod
    def _qe_masking(qe):
        mask = sequence_mask(
            torch.arange(qe.size()[-1] - 1, qe.size()[-1] - qe.size()[-2] - 1, -1).to(qe.device),
            qe.size()[-1],
        )
        mask = ~mask.to(mask.device)
        return mask.to(qe.dtype) * qe


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, rate=0.1, h=16, additional=False, max_seq=512):
        super(EncoderLayer, self).__init__()

        self.d_model = d_model
        self.rga = RelativeGlobalAttention(h=h, d=d_model, max_seq=max_seq, add_emb=additional)

        self.FFN_pre = torch.nn.Linear(self.d_model, self.d_model // 2)
        self.FFN_suf = torch.nn.Linear(self.d_model // 2, self.d_model)

        self.layernorm1 = torch.nn.LayerNorm(self.d_model, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(self.d_model, eps=1e-6)

        self.dropout1 = torch.nn.Dropout(rate)
        self.dropout2 = torch.nn.Dropout(rate)

    def forward(self, x, mask=None, **kwargs):
        attn_out, w = self.rga([x, x, x], mask)
        attn_out = self.dropout1(attn_out)
        out1 = self.layernorm1(attn_out + x)

        ffn_out = F.relu(self.FFN_pre(out1))
        ffn_out = self.FFN_suf(ffn_out)
        ffn_out = self.dropout2(ffn_out)
        out2 = self.layernorm2(out1 + ffn_out)
        return out2, w


class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, rate=0.1, h=16, additional=False, max_seq=512):
        super(DecoderLayer, self).__init__()

        self.d_model = d_model
        self.rga2 = RelativeGlobalAttention(d=d_model, h=h, max_seq=max_seq, add_emb=additional)
        self.rga = RelativeGlobalAttention(d=d_model, h=h, max_seq=max_seq, add_emb=additional)

        self.FFN_pre = torch.nn.Linear(self.d_model, self.d_model // 2)
        self.FFN_suf = torch.nn.Linear(self.d_model // 2, self.d_model)

        self.layernorm1 = torch.nn.LayerNorm(self.d_model, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(self.d_model, eps=1e-6)
        self.layernorm3 = torch.nn.LayerNorm(self.d_model, eps=1e-6)

        self.dropout1 = torch.nn.Dropout(rate)
        self.dropout2 = torch.nn.Dropout(rate)
        self.dropout3 = torch.nn.Dropout(rate)

    def forward(self, x, encode_out, mask=None, lookup_mask=None, w_out=False, **kwargs):

        attn_out, aw1 = self.rga([x, x, x], mask=lookup_mask)
        attn_out = self.dropout1(attn_out)
        out1 = self.layernorm1(attn_out + x)

        if encode_out is None:
            attn_out2, aw2 = self.rga2([out1, out1, out1], mask=mask)
        else:
            attn_out2, aw2 = self.rga2([out1, encode_out, encode_out], mask=mask)
        attn_out2 = self.dropout2(attn_out2)
        attn_out2 = self.layernorm2(out1 + attn_out2)

        ffn_out = F.relu(self.FFN_pre(attn_out2))
        ffn_out = self.FFN_suf(ffn_out)
        ffn_out = self.dropout3(ffn_out)
        out = self.layernorm3(attn_out2 + ffn_out)

        if w_out:
            return out, aw1, aw2
        else:
            return out


class Encoder(torch.nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()

        self.d_model = config.n_embd
        self.num_layers = config.n_layer
        self.max_seq = config.n_ctx
        self.vocab_size = config.meta_vocab_size
        self.dropout_rate = config.dropout_rate

        self.embedding = torch.nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.d_model
        )
        self.enc_layers = torch.nn.ModuleList(
            [
                EncoderLayer(
                    self.d_model,
                    self.dropout_rate,
                    h=self.d_model // 64,
                    additional=False,
                    max_seq=self.max_seq,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.dropout = torch.nn.Dropout(self.dropout_rate)

    def forward(self, x, mask=None):
        weights = []
        # adding embedding and position encoding.
        x = self.embedding(x.to(torch.long))  # (batch_size, input_seq_len, d_model)
        x *= math.sqrt(self.d_model)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x, w = self.enc_layers[i](x, mask)
            weights.append(w)
        return x, weights  # (batch_size, input_seq_len, d_model)


class Decoder(torch.nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()

        self.d_model = config.n_embd
        self.num_layers = config.n_layer
        self.max_seq = config.n_ctx
        self.vocab_size = config.note_vocab_size
        self.dropout_rate = config.dropout_rate

        self.embedding = torch.nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.d_model
        )
        self.pos_encoding = DynamicPositionEmbedding(self.d_model, max_seq=self.max_seq)
        self.dec_layers = torch.nn.ModuleList(
            [
                DecoderLayer(
                    self.d_model,
                    self.dropout_rate,
                    h=self.d_model // 64,
                    additional=False,
                    max_seq=self.max_seq,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.dropout = torch.nn.Dropout(self.dropout_rate)

    def forward(self, x, enc_out, mask=None, lookup_mask=None):
        # adding embedding and position encoding.
        x = self.embedding(x.to(torch.long))  # (batch_size, input_seq_len, d_model)
        x *= math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_out, mask, lookup_mask)
        return x  # (batch_size, input_seq_len, d_model)


class SmoothCrossEntropyLoss(_Loss):
    """패딩 토큰 인덱스를 무시하고 sequence cross_entropy loss를 smoothing_mean으로 계산
    ref)https://arxiv.org/abs/1512.00567
    """

    __constants__ = ["label_smoothing", "vocab_size", "ignore_index", "reduction"]

    def __init__(
        self, label_smoothing, vocab_size, ignore_index=-100, reduction="mean", is_logits=True
    ):
        assert 0.0 <= label_smoothing <= 1.0
        super().__init__(reduction=reduction)

        self.label_smoothing = label_smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.input_is_logits = is_logits

    def forward(self, input, target):
        """
        Args:
            input: [B * T, V]
            target: [B * T]
        Returns:
            cross entropy: [1]
        """
        mask = (target == self.ignore_index).unsqueeze(-1)
        q = F.one_hot(target.long(), self.vocab_size).type(torch.float32)
        u = 1.0 / self.vocab_size
        q_prime = (1.0 - self.label_smoothing) * q + self.label_smoothing * u
        q_prime = q_prime.masked_fill(mask, 0)

        ce = self.cross_entropy_with_logits(q_prime, input)
        if self.reduction == "mean":
            lengths = torch.sum(target != self.ignore_index)
            return ce.sum() / lengths
        elif self.reduction == "sum":
            return ce.sum()
        else:
            raise NotImplementedError

    def cross_entropy_with_logits(self, p, q):
        return -torch.sum(p * (q - q.logsumexp(dim=-1, keepdim=True)), dim=-1)


def beam_search(
    sos_token: int,
    pad_token: int,
    encoded_meta: torch.tensor,
    max_len: int,
    models,
    note_vocab_size: int,
    beam_size: int,
    temperature: int,
) -> List:
    note_in = torch.zeros((1, max_len), dtype=torch.long)
    note_in[0][0] = sos_token
    start = copy.deepcopy(note_in)
    sequences = [[start, 1]]  # start token, score

    for idx in range(max_len - 1):
        res_cadidates: List[List[torch.tensor, float]] = []  # [[note_seq tensor, score]]
        for sample in sequences:
            inputs, score = sample
            out = models(encoded_meta, inputs)
            out = torch.div(out[0], temperature)
            out = out.softmax(-1)
            candidate_token = torch.argsort(out[idx], descending=True)[:note_vocab_size]

            for token in candidate_token:
                if token == pad_token:
                    continue
                tmp_inputs = copy.deepcopy(inputs)
                tmp_score = copy.deepcopy(score)
                new_score = tmp_score * -np.log(out[idx][token].detach())
                tmp_inputs[0][idx + 1] = token
                res = [tmp_inputs, new_score]
                res_cadidates.append(res)
        ordered = sorted(res_cadidates, key=lambda tup: tup[1])  # 점수 기준 정렬
        sequences = ordered[:beam_size]
        if sequences[0][0][0][idx + 1] == 1:
            print("meet eos token")
            return sequences[0][0][0][1 : idx + 2]  # sos token 빼고, eos token 까지 슬라이싱

    return sequences[0][0][0][1:]  # sos token 빼고 끝까지 슬라이싱
