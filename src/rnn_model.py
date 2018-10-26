import torch
import torch.nn as nn
import torch.nn.functional as F

class rnn_model(nn.Module):
    def __init__(self, device, word_embedding=None, n_classes=2, vocab_size=-1):
        super(rnn_model, self).__init__()
        self.device = device
        self.n_classes = n_classes

        if word_embedding:
            self.embedding = word_embedding
            self.input_size= word_embedding.Size()[1]
        else:
            self.input_size=50
            self.embedding = nn.Embedding(vocab_size,self.input_size)
        self.hidden_size = self.input_size
        self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size)
        self.h2o = nn.Linear(self.hidden_size,self.n_classes)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, input,hidden):
        embedded = self.embedding(input).view(-1, 1, self.hidden_size) #one token at a time (may be changed to one seq)
        output = embedded
        output, hidden = self.rnn(output, hidden)
        output = output[-1] #take the last output state
        output = self.h2o(output)
        logprobs = self.logsoftmax(output)
        return logprobs

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)
