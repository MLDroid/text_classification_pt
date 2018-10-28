import torch
import torch.nn as nn
import torch.nn.functional as F

class rnn_model(nn.Module):
    def __init__(self, device, word_embedding=None, n_classes=2, vocab_size=None, use_pretrained_wv=False):
        super(rnn_model, self).__init__()
        self.device = device
        self.n_classes = n_classes

        if use_pretrained_wv:
            self.embedding = self._create_emb_layer(word_embedding, non_trainable=True)
            self.input_size= word_embedding.shape[1]
        else:
            self.input_size=50
            self.embedding = nn.Embedding(vocab_size,self.input_size)
        self.hidden_size = self.input_size
        self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size,dropout=0.4)
        self.h2o = nn.Linear(self.hidden_size,self.n_classes)
        self.logsoftmax = nn.LogSoftmax()
        self.dp20 = nn.Dropout(p=.2)
        self.dp40 = nn.Dropout(p=.4)




    def forward(self, input,hidden,is_train=True):
        embedded = self.embedding(input).view(-1, 1, self.hidden_size) #one token at a time (may be changed to one seq)
        output = embedded
        output, hidden = self.rnn(output, hidden)
        output = output[-1] #take the last output state
        if is_train:
            output = self.dp20(output)
        output = self.h2o(output)
        logprobs = self.logsoftmax(output)
        return logprobs

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)

    def _create_emb_layer(self, weights_matrix, non_trainable=False):
        num_embeddings, embedding_dim = weights_matrix.size()
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight': weights_matrix})
        if non_trainable:
            emb_layer.weight.requires_grad = False

        return emb_layer

if __name__ == '__main__':
    pass