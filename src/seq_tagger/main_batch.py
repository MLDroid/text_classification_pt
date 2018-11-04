import os, json, torch
import utils,rnn_model_batch
from time import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from random import randint,choice
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from collections import Counter,OrderedDict
import data_loader
from pprint import pprint


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bi = True

def test_basline(test_sent_ids, y_test, src_word_id_map, tgt_word_id_map):
    src_id_word_map = {v:k for k,v in src_word_id_map.items()}
    pred_pairs_most_common = []
    pred_pairs_random = []
    for sent in test_sent_ids:
        actual_ip_words = [src_id_word_map[w] for w in sent]
        actual_ip_words = ['~'.join(w.split('~')[:-1]) for w in actual_ip_words]
        actual_ip_word_ids = [tgt_word_id_map.get(id,None) for id in actual_ip_words]
        actual_ip_word_ids = [id for id in actual_ip_word_ids if id]

        if not actual_ip_word_ids:
            actual_ip_word_ids = [-1]

        most_common_word_id = max(set(actual_ip_word_ids), key=actual_ip_word_ids.count)
        rand_word_id = choice(actual_ip_word_ids)

        pred_pairs_most_common.append(most_common_word_id)
        pred_pairs_random.append(rand_word_id)
    results_dict = OrderedDict()
    results_dict['acc_most_common_baseline'] = accuracy_score(y_test, pred_pairs_most_common)
    results_dict['f_most_common_baseline'] = f1_score(y_test, pred_pairs_most_common,average='macro')
    results_dict['acc_random_baseline'] = accuracy_score(y_test, pred_pairs_random)
    results_dict['f_random_baseline'] = f1_score(y_test, pred_pairs_random,average='macro')
    
    return results_dict

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
    # optimizer = optim.SGD(model.parameters(), lr=lr)
    optimizer = optim.Adam(model.parameters(), lr=lr)
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
    all_pairs_fname = '/home/ubuntu/trans_seq/src/seq_classify/seq_tagger/all_pairs_for_tagger.json'
    all_pairs = utils.load_pine_labs_dataset(all_pairs_fname)[:5000]
    all_pairs,removed_stats = utils.remove_low_freq_labels_pairs(all_pairs,threshold=100)
    pprint (removed_stats)

    if load_embedding:
        # embeddings, word_id_map = utils.load_glove_embedding_map(torch_flag=True)
        # print (f'loaded pretrained embedding of shape: ', embeddings.shape)
        exit(-1)
    else:
        print(f'NO pretrained embedding loaded')
        embedding = 100 #if not using pretrained embedding, just give the dimension of embedding here
        src_word_id_map, tgt_word_id_map = utils.get_word_id_maps(all_pairs)
        sents_as_ids, labels = utils.map_word_ids(all_pairs, src_word_id_map,tgt_word_id_map)
        # org_n_samples = len(labels);org_n_classes = len(set(labels))
        # print ('original class distribution')
        # pprint(Counter(labels))
        #
        # low_freq_labels = utils.get_low_freq_labels(labels, threshold=100)
        # sents_as_ids, labels = utils.remove_low_freq_labels(sents_as_ids, labels, low_freq_labels)
        # reduced_n_samples = len(labels);reduced_n_classes = len(set(labels))
        # print('after removing low freq classes: class distribution')
        # pprint(Counter(labels))
        # diff_n_samples = org_n_samples - reduced_n_samples
        # diff_percent = round((diff_n_samples/org_n_samples)*100, 2)
        # print (f'original and reduced number of samples: {org_n_samples} and {reduced_n_samples}, precent of reduction: {diff_percent}')
        # print (f'original and reduced number of classes: {org_n_classes} and {reduced_n_classes}, precent of reduction: '
        #        f'{round(((org_n_classes-reduced_n_classes)/org_n_classes)*100,2)}')

    sents_as_ids = [utils.pad_token_ids(s,src_word_id_map['PAD'],max_len=30) for s in sents_as_ids]



    accs = []
    f_scores = []
    for run in range(1, max_runs+1):
        train_sent_ids, test_sent_ids, y_train, y_test = train_test_split(sents_as_ids, labels, test_size = 0.2, random_state = randint(1,100))

        dataset = data_loader.load_dataset(train_sent_ids, y_train, device)
        train_loader = DataLoader(dataset,batch_size=batch_size, shuffle=True,num_workers=0)
        print (f'sample lengths - train: {len(y_train)} and test: {len(y_test)}')

        # baseline_results = test_basline(test_sent_ids, y_test, src_word_id_map, tgt_word_id_map)
        # print('baseline results: ')
        # pprint(baseline_results)

        model = train(train_loader, embeddings=embedding,
                      n_epochs=100, vocab_size=len(src_word_id_map),
                      lr=0.0001, use_pretrained_wv=load_embedding,
                      n_classes=len(tgt_word_id_map))
        try:
            a,f = test_batch(model, test_sent_ids, y_test)
        except:
            a,f = test_stochastic(model, test_sent_ids, y_test)
        accs.append(a)
        f_scores.append(f)
        print(f'run: {run}, acc: {a}, f1: {f}')
        baseline_results = test_basline(test_sent_ids, y_test, src_word_id_map, tgt_word_id_map)
        print('baseline results: ')
        pprint(baseline_results)

    print(f'average of {max_runs} runs acc: {sum(accs)/len(accs)}, f1: {sum(f_scores)/len(f_scores)}')




if __name__ == '__main__':
    main(load_embedding=False,max_runs=5,batch_size=1)