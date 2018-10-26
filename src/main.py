import os, json, torch
import utils,rnn_model
from time import time
import torch.nn as nn
import torch.optim as optim
from random import shuffle, randint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(model, test_sent_ids, y_test):
    sents_as_ids = [torch.tensor(s, device=device, dtype=torch.long).view(-1, 1) for s in test_sent_ids]
    y_pred = []
    with torch.no_grad():
        for sent in sents_as_ids:
            hidden = model.initHidden()
            logprobs = model.forward(sent, hidden)
            topv, topi = logprobs.topk(1)
            topi = topi.squeeze().detach()  # detach from history as input
            yhat = topi.item()
            y_pred.append(yhat)

    print (f'acc: {accuracy_score(y_test,y_pred)}')


def train(sents_as_ids, labels, embeddings, n_epochs=2, vocab_size=-1, lr=0.01):
    sents_as_ids = [torch.tensor(s,device=device,dtype=torch.long).view(-1,1) for s in sents_as_ids]
    XY = list(zip(sents_as_ids, labels))

    model = rnn_model.rnn_model(device, word_embedding=embeddings, vocab_size=vocab_size)
    model = model.cuda()
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for e in range(1,n_epochs+1):
        shuffle(XY)
        t0 = time()
        losses = []
        for sent,label in XY:
            optimizer.zero_grad()
            hidden = model.initHidden()
            logprobs = model.forward(sent,hidden)
            loss = criterion(logprobs, torch.tensor([label],device=device,dtype=torch.long))
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        epoch_loss = sum(losses)/len(losses)
        epoch_time = round(time()-t0,2)
        print(f'print epoch: {e}, time: {epoch_time}, loss: {epoch_loss}')

    return model



def main():
    sents, labels = utils.load_imdb_dataset()
    tokenized_sents = utils.tokenize(sents)
    # embeddings, word_id_map = utils.load_glove_embedding_map(torch_flag=True)
    word_id_map = utils.get_word_id_map(enrich_vocab=True)
    embeddings=None
    sents_as_ids = utils.convert_word_to_id(tokenized_sents, word_id_map)

    train_sent_ids, test_sent_ids, y_train, y_test = train_test_split(sents_as_ids, labels, test_size = 0.2, random_state = 42)
    model = train(train_sent_ids, y_train, embeddings, n_epochs=50, vocab_size=len(word_id_map),lr=0.1)
    test(model, test_sent_ids, y_test)

if __name__ == '__main__':
    main()