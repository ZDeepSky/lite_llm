
import torch
import torch.nn as nn
import torch.nn.functional as F


class liteEmbedding(nn.Module):
    def __init__(self, vocab_size, hidding_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidding_size)
        # self.pos_embedding = nn.Embedding(max_len, hidding_size)



    def forward(self, input_ids):
        # input_id [batch_size,seq_len]
        #batch_size, seq_len = input_ids.shape()

        token_emb = self.token_embedding(input_ids)
        return token_emb


