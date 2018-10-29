import os, json, torch
import utils,rnn_model_batch
from time import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from random import randint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from collections import Counter
import data_loader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bi = True

def test_batch(model, test_sent_ids, y_test):
    model.eval()
    batch_size = len(test_sent_ids)
    sents_as_ids = torch.tensor(test_sent_ids, device=device, dtype=torch.long)
    with torch.no_grad():
            hidden = model.initHidden(batch_size, bi=bi)
            logprobs = model(sents_as_ids, hidden, is_train=False)
            topv, topi = logprobs.topk(1)
            topi = topi.squeeze().detach()  # detach from history as input
            y_pred = topi.cpu().numpy()
    acc = accuracy_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred,average='macro')
    return acc, f1


def test_stochastic(model, test_sent_ids, y_test):
    model.eval()
    sents_as_ids = [torch.tensor(s, device=device, dtype=torch.long).view(-1, 1) for s in test_sent_ids]
    y_pred = []
    with torch.no_grad():
        for sentindex, sent in enumerate(sents_as_ids):
            hidden = model.initHidden(1, bi=bi)
            logprobs = model.forward(sent, hidden, is_train=False, is_batch=False)
            topv, topi = logprobs.topk(1)
            topi = topi.squeeze().detach()  # detach from history as input
            yhat = topi.item()
            y_pred.append(yhat)
    acc = accuracy_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred,average='macro')
    return acc, f1

def train(train_loader, embeddings, n_epochs=2, vocab_size=-1, lr=0.01, use_pretrained_wv=False, n_classes=2):
    model = rnn_model_batch.rnn_model(device, word_embedding=embeddings, n_classes=n_classes,
                                vocab_size=vocab_size, use_pretrained_wv=use_pretrained_wv, bi=bi)
    model = model.cuda()
    print(f' created RNN (GRU) model: {model}')
    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()
    for e in range(1,n_epochs+1):
        t0 = time()
        losses = []
        for batch_data, batch_labels in train_loader:
            sents = batch_data
            labels = batch_labels
            optimizer.zero_grad()
            hidden = model.initHidden(batch_size=len(labels), bi=bi)
            logprobs = model(sents,hidden, is_batch=True)
            loss = criterion(logprobs, labels)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        epoch_loss = sum(losses)/len(losses)
        epoch_time = round(time()-t0,2)
        print(f'epoch: {e}, time: {epoch_time}, loss: {epoch_loss}')

    return model



def main(load_embedding=False, max_runs=5,batch_size=1):
    sents, labels, n_classes = utils.load_imdb_dataset()
    # sents, labels, n_classes = utils.load_rotten_tomatoes_dataset()
    print(f'distribution of classes: {Counter(labels)}')
    tokenized_sents = utils.tokenize(sents)
    tokenized_sents, labels = utils.remove_empty_tokenizedsents(tokenized_sents, labels)
    if load_embedding:
        embeddings, word_id_map = utils.load_glove_embedding_map(torch_flag=True)
        print (f'loaded pretrained embedding of shape: ', embeddings.shape)
    else:
        print(f'NO pretrained embedding loaded')
        word_id_map = utils.get_word_id_map(enrich_vocab=True)
        embeddings=None

    sents_as_ids = utils.convert_word_to_id(tokenized_sents, word_id_map)
    # sents_as_ids = [utils.pad_token_ids(s,word_id_map['PAD'],max_len=30) for s in sents_as_ids]



    accs = []
    f_scores = []
    for run in range(1, max_runs+1):
        train_sent_ids, test_sent_ids, y_train, y_test = train_test_split(sents_as_ids, labels, test_size = 0.3, random_state = randint(1,100))
        dataset = data_loader.load_dataset(train_sent_ids, y_train, device)
        train_loader = DataLoader(dataset,batch_size=batch_size, shuffle=True,num_workers=0)
        print (f'sample lengths - train: {len(y_train)} and test: {len(y_test)}')

        model = train(train_loader, embeddings, n_epochs=20, vocab_size=len(word_id_map),
                          lr=0.05, use_pretrained_wv=load_embedding, n_classes=n_classes)
        try:
            a,f = test_batch(model, test_sent_ids, y_test)
        except:
            a,f = test_stochastic(model, test_sent_ids, y_test)
        accs.append(a)
        f_scores.append(f)
        print(f'run: {run}, acc: {a}, f1: {f}')
    print(f'average of {max_runs} runs acc: {sum(accs)/len(accs)}, f1: {sum(f_scores)/len(f_scores)}')


if __name__ == '__main__':
    main(load_embedding=False,max_runs=5,batch_size=1)