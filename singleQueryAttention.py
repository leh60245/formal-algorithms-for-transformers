import torch
from torch import nn
from torch.nn import functional as F
import math


class SingleQueryAttention(nn.Module):

    def __init__(self, input_dim, attn_dim, output_dim) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.attn_dim = attn_dim
        self.output_dim = output_dim
        self.query = nn.Linear(input_dim, attn_dim, bias=True)
        self.key = nn.Linear(input_dim, attn_dim, bias=True)
        self.value = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, current_token, context_tokens):
        query = self.query(current_token)
        key = self.key(context_tokens)
        value = self.value(context_tokens)

        attn_score = torch.einsum('ijk,ilk->ijl', [query, key]) / math.sqrt(
            self.attn_dim)
        attn_score = F.softmax(
            attn_score, dim=-1)  # softmax는 마지막 차원인 key의 문장 길이 방향으로 수행된다.
        attn_score = torch.einsum('ijk,ikl->ijl', [attn_score, value])
        return attn_score


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from embedding import TokenEmbedding
    from encoding import PositionalEncoding

    emb_size = 512  # or d_model
    max_vocab_size = 10000
    max_seq_len = 512  # 토큰 문장의 최대 길이
    token_emb = TokenEmbedding(emb_size=emb_size, vocab_size=max_vocab_size)
    pos_enc = PositionalEncoding(d_model=emb_size, max_len=max_seq_len)
    single_query_attention = SingleQueryAttention(input_dim=emb_size,
                                                  attn_dim=emb_size,
                                                  output_dim=emb_size)

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
    print("attention score size:", attn_score.shape)
