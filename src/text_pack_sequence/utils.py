import os,json,nltk
import numpy as np
import torch
import pandas as pd

def load_imdb_dataset():
    fname = '../../data/sentiment labelled sentences/imdb_labelled.txt'
    lines = (l.strip() for l in open(fname))
    sents = []
    labels = []
    for l in lines:
        s,l = l.split('\t')
        sents.append(s)
        labels.append(int(l))

    print (f'loaded {len(sents)} and {len(labels)} from {fname}')
    n_classes = len(set(labels))
    return sents, labels, n_classes

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

def get_word_id_map(word_id_map_fname='../../embedding/word_id_map.json',enrich_vocab=False):

    with open(word_id_map_fname) as fh:
        word_id_map = json.load(fh)

    if enrich_vocab:
        tokens_to_enrich = ['PAD', 'SOS', 'EOS', 'UNK']
        offset = len(tokens_to_enrich)
        word_id_map = {k: v + offset for k, v in word_id_map.items()}
        for i,w in enumerate(tokens_to_enrich):
            word_id_map[w] = i

    return word_id_map

def load_glove_embedding_map(embedding_fname='../../embedding/glove.6B.50d_numpy.txt',
                             enrich_vocab = ['PAD','SOS','EOS','UNK'],
                             torch_flag = False):
    embeddings = np.loadtxt(embedding_fname)
    word_id_map = get_word_id_map(enrich_vocab=False)
    print (f'loaded glove embeddings of shape: {embeddings.shape} and word_id_map of len: {len(word_id_map)}')


    if enrich_vocab:
        tokens_to_enrich = enrich_vocab
        offset = len(tokens_to_enrich)
        word_id_map = {k: v + offset for k, v in word_id_map.items()}

    new_embeddings = []
    dim = embeddings.shape[1]
    for new_word_id, new_word in enumerate(enrich_vocab):
        word_id_map[new_word] = new_word_id
        new_embeddings.append(np.random.randn(dim))
    new_embeddings = np.array(new_embeddings)
    print(f'new words embedding shape: ',new_embeddings.shape)
    embeddings = np.vstack([new_embeddings,embeddings])
    print(f'after enriching the set of word vectors, their shape: ', embeddings.shape)

    if torch_flag:
        embeddings = _get_torch_embeddings (embeddings)
    return embeddings, word_id_map


def convert_word_to_id(tokenized_sents,word_id_map):
    unk_word_id = word_id_map['UNK']
    sents_as_ids = []
    for s_tokens in tokenized_sents:
        s_ids = [word_id_map.get(w,unk_word_id) for w in s_tokens]
        sents_as_ids.append(s_ids)
    return sents_as_ids


def load_rotten_tomatoes_dataset(fname='../../data/rotten_tomatoes_movie_rev/train.tsv'):
    df = pd.read_csv(fname,delimiter='\t')
    print(f'loaded df of shape: {df.shape}')
    gdf = df.groupby('SentenceId')
    sents = []
    labels = []
    for gid, g in gdf:
        s = g.Phrase.iloc[0]
        l = g.Sentiment.iloc[0]
        sents.append(s)
        labels.append(l)
    print(f'loaded {len(sents)} sentences and {len(labels)} labels from data frame')
    n_classes = len(set(labels))
    return sents, labels, n_classes

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

if __name__ == '__main__':
    # sents, labels = load_imdb_dataset()
    sents, labels = load_rotten_tomatoes_dataset()
    tokenized_sents = tokenize(sents)
    tokenized_sents, labels = remove_empty_tokenizedsents(tokenized_sents, labels)
    print_stats_dataset(tokenized_sents)
    # embeddings, word_id_map = load_glove_embedding_map(torch_flag=True)

    word_id_map = get_word_id_map(enrich_vocab=True)
    sents_as_ids = convert_word_to_id(tokenized_sents,word_id_map)