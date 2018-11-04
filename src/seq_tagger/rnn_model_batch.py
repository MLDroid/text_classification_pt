import torch
import torch.nn as nn
import torch.nn.functional as F

class rnn_model(nn.Module):
    def __init__(self, device, word_embedding=None, n_classes=2, vocab_size=None, use_pretrained_wv=False, bi=False):
        super(rnn_model, self).__init__()
        self.device = device
        self.n_classes = n_classes
        self.bi = bi

        if use_pretrained_wv:
            self.embedding = self._create_emb_layer(word_embedding, non_trainable=True)
            self.input_size= word_embedding.shape[1]
        else:
            self.input_size=word_embedding
            self.embedding = nn.Embedding(vocab_size,self.input_size)
        self.hidden_size = self.input_size
        self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size,dropout=0,bidirectional=bi)
        if self.bi:
            self.h2o = nn.Linear(self.hidden_size*2,self.n_classes)
        else:
            self.h2o = nn.Linear(self.hidden_size,self.n_classes)
        self.logsoftmax = nn.LogSoftmax()
        self.softmax = nn.Softmax(dim=1)
        self.dp20 = nn.Dropout(p=.2)
        self.dp40 = nn.Dropout(p=.4)




    def forward(self, input,hidden,is_train=True, is_batch=True):
        if is_batch:
            batch_size = input.shape[0]
            embedded = self.embedding(input).view(-1, batch_size, self.hidden_size) #one token at a time (may be changed to one seq)

        else:
            embedded = self.embedding(input).view(-1, 1,self.hidden_size)  # one token at a time (may be changed to one seq)

        output = embedded
        output, hidden = self.rnn(output, hidden)
        output = output.squeeze(1).transpose()
        if self.bi:
            hidden = hidden.view(-1,1,2*self.hidden_size)
        output = hidden
        if is_train:
            output = self.dp20(output)
        output = self.h2o(output)
        for dim, dim_value in enumerate(output.shape):
            if dim_value == 1:
                dim_to_drop = dim
        output = output.squeeze(dim_to_drop)
        # logprobs = self.logsoftmax(output)
        logprobs = self.softmax(output)
        return logprobs


    def initHidden(self,batch_size=1,bi=False):
        if bi:
            return torch.zeros(2, batch_size, self.hidden_size, device=self.device)
        else:
            return torch.zeros(1, batch_size, self.hidden_size, device=self.device)

    def _create_emb_layer(self, weights_matrix, non_trainable=False):
        num_embeddings, embedding_dim = weights_matrix.size()
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight': weights_matrix})
        if non_trainable:
            emb_layer.weight.requires_grad = False

        return emb_layer

if __name__ == '__main__':
    pass