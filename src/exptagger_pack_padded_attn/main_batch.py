import os, json, torch
import utils,rnn_model_batch
from time import time
import torch.nn as nn
import torch.optim as optim
from random import randint, shuffle, choice
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from collections import Counter,OrderedDict
from pprint import pprint
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bi = False


def test_basline(test_sent_ids, y_test, src_word_id_map, tgt_word_id_map):
    src_id_word_map = {v: k for k, v in src_word_id_map.items()}
    pred_pairs_most_common = []
    pred_pairs_random = []
    for sent in test_sent_ids:
        actual_ip_words = [src_id_word_map[w] for w in sent]
        actual_ip_words = ['~'.join(w.split('~')[:-1]) for w in actual_ip_words]
        actual_ip_word_ids = [tgt_word_id_map.get(id, None) for id in actual_ip_words]
        actual_ip_word_ids = [id for id in actual_ip_word_ids if id]

        if not actual_ip_word_ids:
            actual_ip_word_ids = [-1]

        most_common_word_id = max(set(actual_ip_word_ids), key=actual_ip_word_ids.count)
        rand_word_id = choice(actual_ip_word_ids)

        pred_pairs_most_common.append(most_common_word_id)
        pred_pairs_random.append(rand_word_id)
    results_dict = OrderedDict()
    results_dict['acc_most_common_baseline'] = accuracy_score(y_test, pred_pairs_most_common)
    results_dict['f_most_common_baseline'] = f1_score(y_test, pred_pairs_most_common, average='macro')
    results_dict['p_most_common_baseline'] = precision_score(y_test, pred_pairs_most_common, average='macro')
    results_dict['r_most_common_baseline'] = recall_score(y_test, pred_pairs_most_common, average='macro')

    results_dict['acc_random_baseline'] = accuracy_score(y_test, pred_pairs_random)
    results_dict['f_random_baseline'] = f1_score(y_test, pred_pairs_random, average='macro')
    results_dict['p_random_baseline'] = precision_score(y_test, pred_pairs_random, average='macro')
    results_dict['r_random_baseline'] = recall_score(y_test, pred_pairs_random, average='macro')

    return results_dict

def test_batch(model, samples, labels, pad_id=0, batch_size=1,max_len=64):
    y_pred = []
    y_test = labels
    lens = [len(s) for s in samples]
    lens_samples_labels = list(zip(lens, samples, labels))
    model.eval()

    with torch.no_grad():
        for batch_lens_samples_labels in utils.get_batch(lens_samples_labels, batch_size=batch_size):
            batch_lens_samples_labels = sorted(batch_lens_samples_labels, reverse=True)
            batch_lens, sents, labels = zip(*batch_lens_samples_labels)
            batch_lens = list(batch_lens)


            sents = [utils.pad_token_ids(s, pad_id, max_len=max_len) for s in sents]
            sents = torch.tensor(sents, device=device, dtype=torch.long)
            labels = torch.tensor(labels, device=device, dtype=torch.long)
            hidden = model.initHidden(batch_size=len(labels), bi=bi)
            logprobs = model(sents, batch_lens, hidden)

            topv, topi = logprobs.topk(1)
            topi = topi.squeeze().detach()  # detach from history as input
            y_hat = topi.item()
            y_pred.append(y_hat)
    acc = accuracy_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred,average='macro')
    p = precision_score(y_test,y_pred,average='macro')
    r = recall_score(y_test,y_pred,average='macro')
    return acc, f1, p, r


def train(samples, labels, embeddings, n_epochs=2, vocab_size=-1, lr=0.01, use_pretrained_wv=False,
          pad_id=0, batch_size=1,test_dict=None,test_every=0,use_wt_loss=True,max_len=64):
    lens = [len(s) for s in samples]
    lens_samples_labels = list(zip(lens,samples,labels))
    n_classes = len(set(labels))
    model = rnn_model_batch.rnn_model(device, word_embedding=embeddings, n_classes=n_classes,
                                vocab_size=vocab_size, use_pretrained_wv=use_pretrained_wv, bi=bi,
                                n_layers=1,max_len=max_len)
    model = model.cuda()
    print(f' created RNN (GRU) model: {model}')
    if use_wt_loss:
        label_freq_counts = Counter(labels)
        print('class freq dist: ', sorted(label_freq_counts.items()))
        # total_labels_count = len(labels)
        # label_probs = {label: count / total_labels_count for label, count in label_freq_counts.items()}
        # wts = [label_probs[label] for label in sorted(label_probs.keys())]
        wts = [1./label_freq_counts[label] for label in sorted(label_freq_counts.keys())]
        # wts = [-np.log10(w) for w in wts]
        wts = torch.tensor(wts, device=device, dtype=torch.float)
        print('weights for loss function: ',wts)
        criterion = nn.CrossEntropyLoss(weight=wts,reduction='sum')
    else:
        criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=lr)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.RMSprop(model.parameters(), lr=lr)

    for e in range(1,n_epochs+1):
        model.train()
        t0 = time()
        losses = []
        shuffle(lens_samples_labels)
        for batch_lens_samples_labels in utils.get_batch(lens_samples_labels,batch_size=batch_size):
            batch_lens_samples_labels = sorted(batch_lens_samples_labels, reverse=True)
            batch_lens, sents, labels = zip(*batch_lens_samples_labels)
            batch_lens = list(batch_lens)
            sents = [utils.pad_token_ids(s,pad_id, max_len=max_len) for s in sents]
            sents = torch.tensor(sents, device=device, dtype=torch.long)
            labels = torch.tensor(labels, device=device, dtype=torch.long)
            optimizer.zero_grad()
            hidden = model.initHidden(batch_size=len(labels), bi=bi)
            logprobs = model(sents, batch_lens, hidden)
            loss = criterion(logprobs, labels)
            losses.append(loss.item())
            loss.backward()
            zero_tensor = torch.zeros(100,device=device)
            model.embedding._parameters['weight'].grad[0]=zero_tensor
            optimizer.step()

        epoch_loss = sum(losses)/len(losses)
        epoch_time = round(time()-t0,2)
        print(f'epoch: {e}, time: {epoch_time} sec., loss: {epoch_loss}')
        if e % test_every == 0:
           if test_dict:
               X_test = test_dict['X_test']
               y_test = test_dict['y_test']
               pad_id = test_dict['pad_id']
               a, f, p, r = test_batch(model, X_test, y_test, pad_id, batch_size=1,max_len=max_len)
               print('*'*80)
               print(f'epoch: {e}, on TEST SET acc: {a}, f1: {f}, precision: {p}, recall: {r}')
               print('*' * 80)

    return model



def main(load_embedding=False, max_runs=5,batch_size=1):
    all_pairs_fname = '/home/ubuntu/trans_seq/src/seq_classify/seq_tagger/all_pairs_for_tagger.json'
    all_pairs = utils.load_pine_labs_dataset(all_pairs_fname)[:10000]
    all_pairs = utils.remove_short_seqs(all_pairs,threshold=5)
    all_pairs, removed_stats = utils.remove_low_freq_labels_pairs(all_pairs, threshold=200)
    pprint(removed_stats)

    mean,sd = utils.get_mean_sd_len(all_pairs)
    print(f'mean and std of all sequences: {mean} and {sd}')
    max_len = int(mean+2*sd)
    max_len = 45


    src_word_id_map, tgt_word_id_map = utils.get_word_id_maps(all_pairs)
    if load_embedding:
        emb_fname = '../../embedding/exp_embeddings.txt'
        src_word_id_map_fname = '../../embedding/exp_word_id_map.json'
        src_word_id_map,embeddings = utils.load_word_id_map_embeddings(src_word_id_map_fname, emb_fname)

    else:
        print(f'NO pretrained embedding loaded')
        embeddings = 200

    sents_as_ids, labels = utils.map_word_ids(all_pairs, src_word_id_map, tgt_word_id_map)

    accs = []
    f_scores = []
    p_scores = []
    r_scores = []
    for run in range(1, max_runs+1):
        train_sent_ids, test_sent_ids, y_train, y_test = train_test_split(sents_as_ids, labels, test_size = 0.2,
                                                                          # random_state = randint(1,100))
                                                                          random_state = 42)
        print(f'sample lengths - train: {len(y_train)} and test: {len(y_test)}')

        ################################### base line ##################################################################
        baseline_results = test_basline(test_sent_ids, y_test, src_word_id_map, tgt_word_id_map)
        print('baseline results: ');pprint(baseline_results)


        #################################### get subsequences for improving training ###################################
        print(f'len of train sents and labels BEFORE extracting subseqs: ',len(train_sent_ids),len(y_train))
        train_sent_ids,y_train = utils.get_subseqs(train_sent_ids, y_train, y_test, src_word_id_map,
                                                   tgt_word_id_map,p_select=0.1)
        print(f'len of train sents and labels AFTER extracting subseqs: ', len(train_sent_ids), len(y_train))

        ################################# actual model ################################################################
        test_dict = {'X_test':test_sent_ids,
                     'y_test': y_test,
                     'pad_id':src_word_id_map['PAD'],
                     'batch_size':1}
        model = train(train_sent_ids, y_train, embeddings, n_epochs=1000,
                      vocab_size=len(src_word_id_map),
                      lr=0.0001, use_pretrained_wv=load_embedding,
                      pad_id=src_word_id_map['PAD'],batch_size=batch_size,
                      test_dict=test_dict,test_every=1,
                      use_wt_loss=True,max_len=max_len)

        a, f, p, r = test_batch(model, test_sent_ids, y_test, pad_id=src_word_id_map['PAD'],batch_size=1,max_len=max_len)

        accs.append(a)
        f_scores.append(f)
        p_scores.append(p)
        r_scores.append(r)
        print(f'run: {run}, acc: {a}, f1: {f}, p:{p}, r:{r}')
    print(f'average of {max_runs} runs acc: {sum(accs)/len(accs)}, f1: {sum(f_scores)/len(f_scores)}'
          f'precision: {sum(p_scores)/len(p_scores)}, and recall: {sum(r_scores)/len(r_scores)}')


if __name__ == '__main__':
    print(os.getcwd())
    main(load_embedding=True,max_runs=5,batch_size=1)