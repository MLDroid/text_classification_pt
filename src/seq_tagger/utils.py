import os,json,nltk
import numpy as np
import torch
import pandas as pd
from collections import Counter,OrderedDict



def tokenize(sents):
    tokenized = []
    for i,s in enumerate(sents):
        s = s.lower()
        tokens = nltk.word_tokenize(s)
        tokenized.append(tokens)
    return tokenized

def _get_torch_embeddings (embeddings):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = torch.from_numpy(embeddings).to(device)
    return embeddings

def get_word_id_maps(all_pairs):
    srcs, tgts = zip(*all_pairs)
    src_vocab = sorted(set([item for in_list in srcs for item in in_list])) + ['PAD']
    tgt_vocab = sorted(set([item for item in tgts]))
    src_word_id_map = {w:i for i,w in enumerate(src_vocab)}
    tgt_word_id_map = {w:i for i,w in enumerate(tgt_vocab)}
    return src_word_id_map, tgt_word_id_map

def map_word_ids(all_pairs, src_word_id_map,tgt_word_id_map):
    srcs, tgts = zip(*all_pairs)
    srcs_t = []
    for s in srcs:
        s = [src_word_id_map[i] for i in s]
        srcs_t.append(s)
    srcs = srcs_t
    tgts = [tgt_word_id_map[i] for i in tgts]
    return srcs, tgts

def load_pine_labs_dataset(all_pairs_fname):
    with open(all_pairs_fname) as fh:
        all_pairs = json.load(fh)
    return all_pairs



def pad_token_ids(tokens,pad_token_id, max_len=80):
    cur_len = len(tokens)
    if cur_len > max_len:
        tokens = tokens[-max_len:]
    else:
        tokens = tokens + [pad_token_id] * (max_len - cur_len)
    return tokens


def print_stats_dataset(sents):
    lens = np.array([len(s) for s in sents])
    min_l, max_l, mean_l, std_l = lens.min(), lens.max(), lens.mean(), lens.std()
    print(f'stats: min sent length: {min_l}, max sent length: {max_l}, mean sent length: {mean_l}, std: {std_l}')

def remove_empty_tokenizedsents(tokenized_sents, labels):
    inds_to_rem = [i for i, s in enumerate(tokenized_sents) if len(s) == 0]
    for i in inds_to_rem:
        del tokenized_sents[i]
        del labels[i]
    return tokenized_sents, labels

def get_low_freq_labels(labels, threshold=20):
    c = Counter(labels)
    low_freq_labels = set()
    for k,v in c.items():
        if v <= threshold:
            low_freq_labels.add(k)
    return low_freq_labels


def remove_low_freq_labels_pairs(all_pairs,threshold=100):
    removed_stats = OrderedDict()
    srcs,tgts = zip(*all_pairs)
    org_n_samples = len(srcs)
    org_n_classes = len(set(tgts))
    counts_dict = Counter(tgts)
    tgts_to_removed = set()
    for t,count in counts_dict.items():
        if count < threshold:
            tgts_to_removed.add(t)

    all_pairs_reduced = []
    for pair in all_pairs:
        if pair[1] in tgts_to_removed:
            continue
        else:
            all_pairs_reduced.append(pair)

    srcs, tgts = zip(*all_pairs_reduced)
    red_n_samples = len(srcs)
    red_n_classes = len(set(tgts))
    sampled_red_percent = round(((org_n_samples-red_n_samples)/org_n_samples)*100,2)
    classes_red_percent = round(((org_n_classes-red_n_classes)/org_n_classes)*100,2)
    removed_stats['org_n_samples'] = org_n_samples
    removed_stats['red_n_samples'] = red_n_samples
    removed_stats['org_n_classes'] = org_n_classes
    removed_stats['red_n_classes'] = red_n_classes
    removed_stats['sampled_red_percent'] = sampled_red_percent
    removed_stats['classes_red_percent'] = classes_red_percent
    return all_pairs_reduced, removed_stats



def remove_low_freq_labels(sents_as_ids, labels, low_freq_labels):
    low_freq_labels = set(low_freq_labels)
    s = []
    l = []
    for sent,label in zip(sents_as_ids,labels):
        if label in low_freq_labels:
            continue
        else:
            s.append(sent)
            l.append(label)
    return s,l



if __name__ == '__main__':
    # sents, labels = load_imdb_dataset()
    sents, labels = load_rotten_tomatoes_dataset()
    tokenized_sents = tokenize(sents)
    tokenized_sents, labels = remove_empty_tokenizedsents(tokenized_sents, labels)
    print_stats_dataset(tokenized_sents)
    # embeddings, word_id_map = load_glove_embedding_map(torch_flag=True)

    word_id_map = get_word_id_map(enrich_vocab=True)
    sents_as_ids = convert_word_to_id(tokenized_sents,word_id_map)