import torch
import torch.nn as nn
import torch.nn.functional as F

class rnn_model(nn.Module):
    def __init__(self, device, word_embedding=None, n_classes=2, vocab_size=None, use_pretrained_wv=False, bi=False,
                 n_layers=2,max_len=0):
        super(rnn_model, self).__init__()
        self.device = device
        self.n_classes = n_classes
        self.bi = bi
        self.dir = 2 if bi else 1
        self.n_layers = n_layers


        if use_pretrained_wv:
            self.embedding = self._create_emb_layer(word_embedding, non_trainable=False)
            self.input_size= word_embedding.shape[1]
        else:
            self.input_size=word_embedding
            self.embedding = nn.Embedding(vocab_size,self.input_size)
        self.hidden_size = self.input_size
        self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size,bidirectional=bi,
                          batch_first=False,num_layers=self.n_layers,dropout=0.4)

        self.logsoftmax = nn.LogSoftmax()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

        self.dp20 = nn.Dropout(p=.2)
        self.dp40 = nn.Dropout(p=.4)
        self.dp60 = nn.Dropout(p=.6)


        self.fc1 = nn.Linear(100*self.dir, 100,bias=False)
        self.fc2 = nn.Linear(4500, self.n_classes)

        # self.fc1 = nn.Linear(self.hidden_size * self.dir * max_len, 100)
        # self.fc2 = nn.Linear(100, self.n_classes)
        # self.bn1 = nn.BatchNorm1d(self.dir * self.hidden_size)
        # self.bn2 = nn.BatchNorm1d(100)

        # self.fc1 = nn.Linear(self.hidden_size * self.dir * self.n_layers, 100)
        # self.fc2 = nn.Linear(100, self.n_classes)
        # self.bn1 = nn.BatchNorm1d(self.dir * self.n_layers * self.hidden_size)
        # self.bn2 = nn.BatchNorm1d(100)

    def _supress_to_zeros(self,rnn_output, valid_length):
        full_tensor_len = rnn_output.shape[0]
        hidden_dims = rnn_output.shape[1]
        len_to_supress = full_tensor_len - valid_length
        lower_tensor = torch.zeros((len_to_supress,hidden_dims),device=self.device)
        upper_tensor = rnn_output[:valid_length,:]
        supressed_tensor = torch.cat((upper_tensor,lower_tensor),0)
        return supressed_tensor




    def forward(self, input, lens, hidden):
        batch_size = input.shape[0]
        valid_len = lens[0]
        embedded = self.embedding(input)
        embedded = embedded.transpose(0,1)
        output, hidden = self.rnn(embedded, hidden)

        # output = output.contiguous()
        output = output.squeeze(1)
        if valid_len < 45:
            output = self._supress_to_zeros(output,valid_len)
        output = self.dp20(output)
        # output = self.bn1(output)

        output = self.fc1(output)
        output = output.contiguous()
        output = output.view(1,-1)
        output = self.relu(output)
        # output = self.bn2(output)

        output = self.fc2(output)
        # output = self.relu(output)
        output = self.dp20(output)
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