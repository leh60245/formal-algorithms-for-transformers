import torch
from torch import nn
from torch.nn import functional as F
import math


class Attention(nn.Module):

    def __init__(self, input_dim, attn_dim, output_dim) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.attn_dim = attn_dim
        self.output_dim = output_dim
        self.query = nn.Linear(input_dim, attn_dim, bias=True)
        self.key = nn.Linear(input_dim, attn_dim, bias=True)
        self.value = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, primary, context_sequence, mask=None):
        query = self.query(primary)
        key = self.key(context_sequence)
        value = self.value(context_sequence)

        score = torch.einsum('ijk,ilk->ijl', [query, key]) / math.sqrt(
            self.attn_dim)

        # attention이 bidirection이라면 필요없다.
        # attention이 unidirection이라면 mask를 씌워야한다.
        # 오류) 논문에 나오는 tz <= tx의 의미를 정확하게 모르겠다.
        if mask is not None:
            score += (mask * -1e9)

        attn_score = F.softmax(
            score, dim=-1)  # softmax는 마지막 차원인 key의 문장 길이 방향으로 수행된다.
        output = torch.einsum('ijk,ikl->ijl', [attn_score, value])
        return output


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from embedding import TokenEmbedding
    from encoding import PositionalEncoding

    emb_size = 512  # or d_model
    max_vocab_size = 10000
    max_seq_len = 512  # 토큰 문장의 최대 길이
    token_emb = TokenEmbedding(emb_size=emb_size, vocab_size=max_vocab_size)
    pos_enc = PositionalEncoding(d_model=emb_size, max_len=max_seq_len)
    attention = Attention(input_dim=emb_size,
                          attn_dim=emb_size,
                          output_dim=emb_size)

    # 예제 생성
    vocab_size = 100
    seq_len = 275
    batch_size = 32
    current_token = torch.randint(0, vocab_size, size=(batch_size, seq_len))
    context_tokens = torch.randint(0, vocab_size, size=(batch_size, seq_len))

    # token embedding -> position encoding -> self query attention
    current_token_embedding = token_emb(current_token)
    context_tokens_embedding = token_emb(context_tokens)
    current_position_encoding = pos_enc(current_token_embedding)
    context_position_encoding = pos_enc(context_tokens_embedding)
    attn_score = attention(current_position_encoding,
                           context_position_encoding)
    print("attention score size:", attn_score.shape)
