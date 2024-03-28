import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor

import math


class singleHeadAttention(nn.Module):

    def __init__(
        self,
        d_x: int,
        d_z: int,
        d_attn: int,
        d_out: int,
        bias: bool = True,
        dropout_proba: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_x = d_x
        self.d_z = d_z
        self.d_attn = d_attn
        self.d_out = d_out
        self.bias = bias
        self.dropout_proba = dropout_proba
        self.scale = 1 / math.sqrt(d_attn)

        self.query = nn.Linear(d_x, d_attn, bias=bias)
        self.key = nn.Linear(d_z, d_attn, bias=bias)
        self.value = nn.Linear(d_z, d_out, bias=bias)
        self.drop = nn.Dropout(dropout_proba)

    def forward(self, x: Tensor, z: Tensor, mask: Tensor = None):
        assert x.dim() == 3
        assert z.dim() == 3

        b_x, l_x, d_x = x.shape
        b_z, l_z, d_z = z.shape
        assert b_x == b_z
        b = b_x
        assert d_x == self.d_x
        assert d_z == self.d_z
        assert mask.shape == (b, l_x, l_z)

        q = self.query(x)
        k = self.key(z)
        v = self.value(z)
        assert q.shape == (b, l_x, self.d_attn)
        assert k.shape == (b, l_z, self.d_attn)
        assert v.shape == (b, l_z, self.d_out)

        score = torch.einsum("b i k, b j k -> b i j", [q, k]) * self.scale
        score = score.masked_fill(~mask.to(torch.bool),
                                  torch.finfo(score.dtype).min)
        # multiplying by mask below is not required but ensures
        # attention is 0 where mask is 0
        attention = torch.softmax(score, dim=-1) * mask
        assert score.shape == attention.shape == (b, l_x, l_z)

        attention = self.drop(attention)

        vtilde = torch.einsum('b i j, b j k -> b i k', [attention, v])
        assert vtilde.shape == (b, l_x, self.d_out)

        return {
            "q": q,
            "k": k,
            "v": v,
            "score": score,
            "attention": attention,
            "vtilde": vtilde,
        }

    def extra_repr(self):
        return "d_x={}, d_z={}, d_attn={}, d_out={}, bias={}".format(
            self.d_x, self.d_z, self.d_attn, self.d_out, self.bias)


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from embedding import TokenEmbedding
    from encoding import PositionalEncoding

    emb_size = 512  # or d_model
    max_vocab_size = 10000
    max_seq_len = 512  # 토큰 문장의 최대 길이
    token_emb = TokenEmbedding(emb_size=emb_size, vocab_size=max_vocab_size)
    pos_enc = PositionalEncoding(d_model=emb_size, max_len=max_seq_len)
    attention = singleHeadAttention(d_x=emb_size,
                                    d_z=emb_size,
                                    d_attn=emb_size,
                                    d_out=emb_size)

    # 예제 생성
    vocab_size = 100
    seq_len = 275
    batch_size = 32
    current_token = torch.randint(0, vocab_size, size=(batch_size, seq_len))
    context_tokens = torch.randint(0, vocab_size, size=(batch_size, seq_len))
    mask = torch.randint(0, 2, size=(batch_size, seq_len, seq_len))

    # token embedding -> position encoding -> single head attention
    current_token_embedding = token_emb(current_token)
    context_tokens_embedding = token_emb(context_tokens)
    current_position_encoding = pos_enc(current_token_embedding)
    context_position_encoding = pos_enc(context_tokens_embedding)
    attn_score = attention(current_position_encoding,
                           context_position_encoding,
                           mask)
    for key, value in attn_score.items():
        print("{}: {}.".format(key.capitalize(), value))
