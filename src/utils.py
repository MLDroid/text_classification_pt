import os,json,nltk
import numpy as np
import torch

def load_imdb_dataset():
    fname = '../data/sentiment labelled sentences/imdb_labelled.txt'
    lines = (l.strip() for l in open(fname))
    sents = []
    labels = []
    for l in lines:
        s,l = l.split('\t')
        sents.append(s)
        labels.append(int(l))

    print (f'loaded {len(sents)} and {len(labels)} from {fname}')
    return sents, labels

def tokenize(sents):
    tokenized = []
    for s in sents:
        s = s.lower()
        tokens = nltk.word_tokenize(s)
        tokenized.append(tokens)
    return tokenized

def _get_torch_embeddings (embeddings):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = torch.from_numpy(embeddings).to(device)
    return embeddings

def get_word_id_map(word_id_map_fname='../embedding/word_id_map.json',enrich_vocab=False):
    with open(word_id_map_fname) as fh:
        word_id_map = json.load(fh)

    if enrich_vocab:
        max_id = max(word_id_map.values())
        for i,w in enumerate(['SOS','EOS','PAD','UNK']):
            word_id_map[w] = max_id + i + 1

    return word_id_map

def load_glove_embedding_map(embedding_fname='../embedding/glove.6B.50d_numpy.txt',
                             enrich_vocab = ['SOS','EOS','PAD','UNK'],
                             torch_flag = False):
    embeddings = np.loadtxt(embedding_fname)
    word_id_map = get_word_id_map(enrich_vocab=False)
    print (f'loaded glove embeddings of shape: {embeddings.shape} and word_id_map of len: {len(word_id_map)}')

    max_id = max(word_id_map.values())
    new_embeddings = []
    dim = embeddings.shape[1]
    for new_word_id, new_word in enumerate(enrich_vocab):
        new_word_id = max_id + new_word_id + 1
        word_id_map[new_word] = new_word_id
        new_embeddings.append(np.random.randn(dim))
    new_embeddings = np.array(new_embeddings)
    print(f'new words embedding shape: ',new_embeddings.shape)
    embeddings = np.vstack([embeddings,new_embeddings])
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





if __name__ == '__main__':
    sents, labels = load_imdb_dataset()
    tokenized_sents = tokenize(sents)
    # embeddings, word_id_map = load_glove_embedding_map(torch_flag=True)




    sents_as_ids = convert_word_to_id(tokenized_sents,word_id_map)