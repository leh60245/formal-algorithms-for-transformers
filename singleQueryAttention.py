import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor

import math


class SingleQueryAttention(nn.Module):

    def __init__(self,
                 d_x: int,
                 d_z: int,
                 d_attn: int,
                 d_out: int,
                 bias: bool = True) -> None:
        super().__init__()
        self.d_x = d_x
        self.d_z = d_z
        self.d_attn = d_attn
        self.d_out = d_out
        self.bias = bias
        self.scale = 1 / math.sqrt(d_attn)

        self.query = nn.Linear(d_x, d_attn, bias=bias)
        self.key = nn.Linear(d_z, d_attn, bias=bias)
        self.value = nn.Linear(d_z, d_out, bias=bias)

    def forward(self, x1: Tensor, z: Tensor):
        assert x1.dim() == 3
        assert z.dim() == 3

        _, _, d_x = x1.shape
        _, l_z, d_z = z.shape
        assert d_x == self.d_x
        assert d_z == self.d_z

        q = self.query(x1)
        k = self.key(z)
        v = self.value(z)
        # q의 경우 single query라 2번째 차원을 1로 hard coding 함
        assert q.shape == (_, 1, self.d_attn)
        assert k.shape == (_, l_z, self.d_attn)
        assert v.shape == (_, l_z, self.d_out)

        score = torch.einsum("b i k, b j k -> b i j", [q, k]) * self.scale
        attention = torch.softmax(score, dim=-1)
        assert score.shape == attention.shape == (_, 1, l_z)

        vtilde = torch.einsum('b i j, b j k -> b i k', [attention, v])

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
    single_query_attention = SingleQueryAttention(d_x=emb_size,
                                                  d_z=emb_size,
                                                  d_attn=emb_size,
                                                  d_out=emb_size)

    # 예제 생성
    vocab_size = 100
    seq_len = 275
    batch_size = 32
    current_token = torch.randint(0, vocab_size, size=(batch_size, 1))
    context_tokens = torch.randint(0, vocab_size, size=(batch_size, seq_len))
    
    # token embedding -> position encoding -> self query attention
    current_token_embedding = token_emb(current_token)
    context_tokens_embedding = token_emb(context_tokens)
    current_position_encoding = pos_enc(current_token_embedding)
    context_position_encoding = pos_enc(context_tokens_embedding)
    attn_score = single_query_attention(current_position_encoding,
                                        context_position_encoding)
    for key, value in attn_score.items():
        print("{}: {}.".format(key.capitalize(), value))