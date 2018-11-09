import os,json,nltk,random
import numpy as np
import torch
import pandas as pd
from collections import OrderedDict,Counter,defaultdict

def load_pine_labs_dataset(all_pairs_fname):
    with open(all_pairs_fname) as fh:
        all_pairs = json.load(fh)
    return all_pairs

def remove_short_seqs(all_pairs,threshold=5):
    selected_pairs = [p for p in all_pairs if len(p[0])>=threshold]
    num_pairs_rem = len(all_pairs) - len(selected_pairs)
    percent_reduction = round(num_pairs_rem/len(all_pairs)*100,2)
    print(f'removed a total of {num_pairs_rem} sequeneces as they are less than {threshold} tokens long')
    print(f'this reduction amounts to removing {percent_reduction}% of samples')

    return selected_pairs

def get_mean_sd_len(pairs,is_tgt_sequence=False):
    srcs,tgts = zip(*pairs)
    src_lens = [len(s) for s in srcs]
    src_lens = np.array(src_lens)
    src_m,src_sd = src_lens.mean(), src_lens.std()
    if is_tgt_sequence:
        tgt_lens = [len(s) for s in tgts]
        tgt_lens = np.array(tgt_lens)
        tgt_m, tgt_sd = tgt_lens.mean(), tgt_lens.std()
        return (src_m,src_sd),(tgt_m,tgt_sd)
    else:
        return (src_m,src_sd)

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


def get_word_id_maps(all_pairs):
    srcs, tgts = zip(*all_pairs)
    src_vocab = ['PAD'] + sorted(set([item for in_list in srcs for item in in_list]))
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


def _get_torch_embeddings (embeddings):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = torch.from_numpy(embeddings).to(device)
    return embeddings


def convert_word_to_id(tokenized_sents,word_id_map):
    unk_word_id = word_id_map['UNK']
    sents_as_ids = []
    for s_tokens in tokenized_sents:
        s_ids = [word_id_map.get(w,unk_word_id) for w in s_tokens]
        sents_as_ids.append(s_ids)
    return sents_as_ids


def pad_token_ids(tokens,pad_token_id, max_len=80):
    cur_len = len(tokens)
    if cur_len > max_len:
        tokens = tokens[:max_len]
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


def get_batch(samples, batch_size=1):
    for i in range(0, len(samples), batch_size):
        yield samples[i:i + batch_size]


def _get_torch_embeddings (embeddings):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = torch.from_numpy(embeddings).to(device)
    return embeddings

def load_word_id_map_embeddings(src_word_id_map_fname, emb_fname, enrich_vocab = ['PAD'], torch_flag = True):
    embeddings = np.loadtxt(emb_fname)
    with open(src_word_id_map_fname,'r') as fh:
        src_word_id_map = json.load(fh)
    print(f'loaded word-id map of size: {len(src_word_id_map)} and embeddings of shape: {embeddings.shape}')

    if enrich_vocab:
        dim = embeddings.shape[1]
        tokens_to_enrich = enrich_vocab
        print(f'about to enrich vocab with the following tokens: {tokens_to_enrich}')
        num_new_tokens = len(tokens_to_enrich)
        src_word_id_map = {word:index+num_new_tokens for word,index in src_word_id_map.items()}
        for new_word_index, new_word in enumerate(tokens_to_enrich):
            src_word_id_map[new_word] = new_word_index
        new_embeds = []
        for t in tokens_to_enrich:
            vec = np.zeros(dim)
            new_embeds.append(vec)
        new_embeds = np.array(new_embeds)
        embeddings = np.vstack([new_embeds,embeddings])
        print(f'after enriching, word-id map size: {len(src_word_id_map)} and embeddings of shape: {embeddings.shape}')

    if torch_flag:
        embeddings = _get_torch_embeddings(embeddings)

    return src_word_id_map, embeddings


def _get_all_subseqs(sent,min_len=5,p_select=0.2):
    subseqs_pairs = []
    if len(sent) < min_len:
        return subseqs_pairs
    else:
        min_index_to_iter = min_len
        max_index_to_iter = len(sent)-2
        for cutoff in range(min_index_to_iter,max_index_to_iter):
            if random.random() <= p_select:
                subseq = sent[:cutoff]
                label = '~'.join(sent[cutoff].split('~')[:-1])
                pair = (subseq,label)
                subseqs_pairs.append(pair)
        return subseqs_pairs

def get_subseqs(train_sent_ids, y_train, y_test, src_word_id_map, tgt_word_id_map, p_select=0.2):
    src_id_word_map = {v:k for k,v in src_word_id_map.items()}
    tgt_id_word_map = {v:k for k,v in tgt_word_id_map.items()}
    train_sents = [[src_id_word_map[w] for w in sent] for sent in train_sent_ids]
    test_labels = set([tgt_id_word_map[l] for l in y_test])


    ext_sents_labels_ids = []
    for sent in train_sents:
        all_subseqs = _get_all_subseqs(sent,p_select=p_select)
        all_subseqs = [pair for pair in all_subseqs if pair[1] in test_labels]
        all_subseqs_ids = [([src_word_id_map[w] for w in pair[0]],tgt_word_id_map[pair[1]]) for pair in all_subseqs]
        ext_sents_labels_ids.extend(all_subseqs_ids)

    ext_train_sent_ids, ext_y_train = zip(*ext_sents_labels_ids)
    train_sent_ids.extend(ext_train_sent_ids)
    y_train.extend(ext_y_train)
    return train_sent_ids, y_train

def get_modes(sents):
    modes = [max(set(s), key=s.count) for s in sents]
    return modes


if __name__ == '__main__':
    pass