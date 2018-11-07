import torch
import torch.nn as nn
import torch.nn.functional as F

class rnn_model(nn.Module):
    def __init__(self, device, word_embedding=None, n_classes=2, vocab_size=None, use_pretrained_wv=False, bi=False,
                 n_layers=2, mode_concat=False):
        super(rnn_model, self).__init__()
        self.device = device
        self.n_classes = n_classes
        self.bi = bi
        self.dir = 2 if bi else 1
        self.n_layers = n_layers
        self.mode_concat = 2 if mode_concat else 1

        if use_pretrained_wv:
            self.embedding = self._create_emb_layer(word_embedding, non_trainable=False)
            self.input_size= word_embedding.shape[1]
        else:
            self.input_size=word_embedding
            self.embedding = nn.Embedding(vocab_size,self.input_size)
        self.hidden_size = self.input_size
        self.rnn = nn.GRU(input_size=self.input_size * self.mode_concat, hidden_size=self.hidden_size,bidirectional=bi,
                          batch_first=True,num_layers=self.n_layers,dropout=0.4)

        self.logsoftmax = nn.LogSoftmax()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

        self.dp20 = nn.Dropout(p=.2)
        self.dp40 = nn.Dropout(p=.4)
        self.dp60 = nn.Dropout(p=.6)

        self.fc1 = nn.Linear(self.hidden_size * self.dir, 100)
        self.fc2 = nn.Linear(100, self.n_classes)
        self.bn1 = nn.BatchNorm1d(self.dir * self.hidden_size)
        self.bn2 = nn.BatchNorm1d(100)

        # self.fc1 = nn.Linear(self.hidden_size * self.dir * self.n_layers, 100)
        # self.fc2 = nn.Linear(100, self.n_classes)
        # self.bn1 = nn.BatchNorm1d(self.dir * self.n_layers * self.hidden_size)
        # self.bn2 = nn.BatchNorm1d(100)




    def forward(self, input,lens, hidden, batch_first=True, modes=None):
        batch_size = input.shape[0]
        embedded = self.embedding(input)
        if modes is not None:
            mode_embedded = self.embedding(modes).view(batch_size,1,-1)
            ones = torch.ones(embedded.shape[0], embedded.shape[1], 1, device=self.device)
            mode_embedded = torch.add(ones,mode_embedded)
            embedded = torch.cat((mode_embedded,embedded),2)
            embedded = self.dp20(embedded)
        output = torch.nn.utils.rnn.pack_padded_sequence(embedded, lens, batch_first=True)
        output, hidden = self.rnn(output, hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)


        # output = hidden.view(batch_size, -1)

        # lens = [lens[0]]*len(lens)
        idx = (torch.tensor(lens,dtype=torch.long) - 1).view(-1, 1).expand(len(lens), output.size(2))
        # idx = torch.tensor(lens,dtype=torch.long).view(-1, 1).expand(len(lens), output.size(2))
        # idx = (torch.tensor(lens,dtype=torch.long)).view(-1, 1).expand(len(lens), output.size(2))
        time_dimension = 1 if batch_first else 0
        idx = idx.unsqueeze(time_dimension)

        idx = idx.cuda(output.data.get_device())
        last_output = output.gather(time_dimension, idx).squeeze(time_dimension)
        output = last_output

        output = self.dp40(output)
        # output = self.bn1(output)

        output = self.fc1(output)
        output = self.relu(output)
        # output = self.bn2(output)

        output = self.fc2(output)
        output = self.relu(output)
        output = self.dp60(output)
        # logprobs = self.logsoftmax(output)
        logprobs = self.softmax(output)
        return logprobs


    def initHidden(self,batch_size=1,bi=False):
        return torch.zeros(self.dir*self.n_layers, batch_size, self.hidden_size, device=self.device)

    def _create_emb_layer(self, weights_matrix, non_trainable=False):
        num_embeddings, embedding_dim = weights_matrix.shape
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight': weights_matrix})
        if non_trainable:
            emb_layer.weight.requires_grad = False

        return emb_layer

if __name__ == '__main__':
    pass