import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):

    def __init__(self, emb_size: int, vocab_size: int):
        super().__init__()
        self.emb_size = emb_size
        self.embedding = nn.Embedding(vocab_size, emb_size)

    def forward(self, tokens):
        return self.embedding(tokens)


if __name__ == "__main__":
    emb_size = 512  # 벡터 표현 / 토큰 임베딩
    vocab_size = 10000  # 사전의 크기
    token_emb = TokenEmbedding(emb_size=emb_size, vocab_size=vocab_size)

    vocab_size = 100
    seq_len = 128  # 입력되는 문장의 길이
    batch_size = 32
    idx = torch.randint(0, vocab_size, size=(batch_size, seq_len))
    print("idx size:", idx.shape)

    idx_embedding = token_emb(idx)
    print("index embedding size:", idx_embedding.shape)
