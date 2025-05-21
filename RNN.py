import torch
import torch.nn as nn
import torch.utils
import torch.utils.data

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, seq_length):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.seq_legth = seq_length

        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.embed(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)
