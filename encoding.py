import math
import torch
from torch import nn
from matplotlib import pyplot as plt

class PositionalEncoding(nn.Module):
    """
    임베딩 벡터의 크기가 d_model이고, token의 개수가 max_len인 임베딩 벡터를 위한 포지셔닝 인코더
    max_len: input sequenc lenght, which is at most some fixed number l-max. 
    """

    def __init__(self, d_model, max_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # torch.arange(n)는 1부터 n까지 숫자를 원소로 가지는 배열을 만든다.
        # unsqueeze(n)을 하면 1인 차원을 생성해주며 생성되는 차원의 위치는 n번째 차원이 된다.
        position = torch.arange(max_len).unsqueeze(1)

        # torch.arange(0, d_model, 2)로 2개씩 증가하여 0부터 d_model까지 총 길이가 64인 array를 만든다.
        # exp(idx * (ln(10,000) / 50))
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # 차원이 (50, 1, 128)인 0으로만 된 array을 우선 만든다.
        # 짝수번째 index에는 sin, 홀수번째 index에는 cos함수를 사용한다.
        pe = torch.zeros(max_len, 1, d_model)  # (50, 1, 128)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        # 모델이 매개변수를 갱신하지 않도록 설정한다.
        self.register_buffer("pe", pe)

    def forward(self, x):
        x += self.pe[:x.size(0)]
        return self.dropout(x)


if __name__ == "__main__":
    from embedding import TokenEmbedding
    emb_size = 512
    vocab_size = 10000
    max_len = 512   # 토큰 문장의 최대 길이
    token_emb = TokenEmbedding(emb_size=emb_size, vocab_size=vocab_size)
    pos_enc = PositionalEncoding(d_model=emb_size, max_len=max_len)
    
    # 예제 생성
    vocab_size = 100
    batch_size = 32
    seq_len = 128
    idx = torch.randint(0, vocab_size, size=(batch_size, seq_len))
    print("idx size:", idx.shape)
    
    idx_embedding = token_emb(idx)
    pos_embedding = pos_enc(idx_embedding)
    print("pos embedding:", pos_embedding.shape)
    
    # embedding plot 생성
    plt.pcolormesh(pos_enc.pe.numpy().squeeze(), cmap="RdBu")
    plt.xlabel("Embedding Dimension")
    plt.xlim((0, 128))
    plt.ylabel("Position")
    plt.colorbar()
    plt.savefig("Embedding plot")