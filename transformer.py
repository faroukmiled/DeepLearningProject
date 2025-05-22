import torch
import numpy as np
import torch.nn as nn
import torch.utils
from  torch.utils.data import random_split,DataLoader,TensorDataset
import time
import torch.nn.functional as F
from RNN import *
from LSTM import *
from generate import *

class PositionalEncoding(nn.Module):

    def __init__(self, emb_dim, max_len = 1000):
        super().__init__()
        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0,max_len).unsqueeze(1)
        # Scaling term for the positional encoding done with Sine and Cosine
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * (-torch.log(torch.tensor(10000.0))/emb_dim))

        # Add sine and cosine to even and odd positions respectively
        pe[:, 0::2] = torch.sin(position * div_term)
        if emb_dim % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])  # Skip last value if mismatch


        self.pe = pe.unsqueeze(0)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x
class CharTransformer(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, n_heads=4, n_layers = 2, ff_dim = 512, dropout = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.pos_enc = PositionalEncoding(emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, dim_feedforward=ff_dim, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(emb_dim, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x) # (B,T,D)
        x = self.pos_enc(x) # Add positional encoding into the embedding
        x = x.permute(1,0,2) # Switch into shape (T, B, D) to train the transformer

        seq_len = x.size(0)
        mask = torch.triu(torch.ones(seq_len,seq_len, device=x.device), diagonal=1).bool() # causal mask
        out = self.transformer(x, mask=mask)
        out = out.permute(1,0,2) # Change the shape back to (B,T,D)
        logits = self.fc_out(out) #(B,T,V)
        return logits  