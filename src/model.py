import torch
import torch.nn as nn

class LSTMGenerator(nn.Module):

    def __init__(self, tokens, char2int, hidden_size, num_layers, dropout, lr):
        super(LSTMGenerator, self).__init__()
        self.input_size = len(tokens)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr

        self.tokens = tokens
        self.char2int = char2int

        self.lstm = nn.LSTM(
            input_size=len(tokens),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(in_features=hidden_size, out_features=len(tokens))
