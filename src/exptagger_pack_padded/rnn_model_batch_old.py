import torch
import torch.nn as nn
import torch.nn.functional as F

class rnn_model(nn.Module):
    def __init__(self, device, word_embedding=None, n_classes=2, vocab_size=None, use_pretrained_wv=False, bi=False):
        super(rnn_model, self).__init__()
        self.device = device
        self.n_classes = n_classes
        self.bi = bi
        self.dir = 2 if bi else 1
        self.n_layers = 1

        if use_pretrained_wv:
            self.embedding = self._create_emb_layer(word_embedding, non_trainable=False)
            self.input_size= word_embedding.shape[1]
        else:
            self.input_size=word_embedding
            self.embedding = nn.Embedding(vocab_size,self.input_size)
        self.hidden_size = self.input_size
        self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size,bidirectional=bi)
        if self.bi:
            self.h2o = nn.Linear(self.hidden_size*2,self.n_classes)
        else:
            self.h2o = nn.Linear(self.hidden_size,self.n_classes)
        self.logsoftmax = nn.LogSoftmax()
        self.softmax = nn.Softmax(dim=1)
        self.dp20 = nn.Dropout(p=.2)
        self.dp40 = nn.Dropout(p=.4)
        self.bn2 = nn.BatchNorm1d(self.hidden_size)




    def forward(self, input,lens, hidden):
        batch_size = input.shape[0]
        embedded = self.embedding(input)
        output = torch.nn.utils.rnn.pack_padded_sequence(embedded, lens, batch_first=True)
        output, hidden = self.rnn(output, hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = hidden.view(batch_size, -1)

        output = self.dp20(output)
        output = self.bn2(output)

        output = self.h2o(output)
        # logprobs = self.logsoftmax(output)
        logprobs = self.softmax(output)
        return logprobs


    def initHidden(self,batch_size=1,bi=False):
        if bi:
            return torch.zeros(2, batch_size, self.hidden_size, device=self.device)
        else:
            return torch.zeros(1, batch_size, self.hidden_size, device=self.device)

    def _create_emb_layer(self, weights_matrix, non_trainable=False):
        num_embeddings, embedding_dim = weights_matrix.shape
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight': weights_matrix})
        if non_trainable:
            emb_layer.weight.requires_grad = False

        return emb_layer

if __name__ == '__main__':
    pass