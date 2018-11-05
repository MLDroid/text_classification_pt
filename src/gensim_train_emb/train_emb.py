import gensim,json,glob,os
import argparse
import psutil
from time import time
import numpy as np


def load_pine_labs_dataset(all_pairs_fname):
    with open(all_pairs_fname) as fh:
        all_pairs = json.load(fh)
    return all_pairs

def get_vocab_embeddings(model):
    vocab = sorted(model.wv.vocab.keys())
    vecs = []
    for word in vocab:
        v = model[word]
        vecs.append(v)
    vecs = np.array(vecs)
    return vocab, vecs

def main(end_index=None, emb_size=100, span=3, epochs=100, save_gensim_model=False):
    embedding_folder = '../../embedding'
    gensim_model_fname = 'w2v_model'
    word_id_map_fname = os.path.join(embedding_folder, 'exp_word_id_map.json')
    word_vectors_fname = os.path.join(embedding_folder,'exp_embeddings.txt')

    n_cpus = psutil.cpu_count()
    print(f'found {n_cpus} CPUs in the machine')
    all_pairs_fname = '/home/ubuntu/trans_seq/src/seq_classify/seq_tagger/all_pairs_for_tagger.json'
    all_pairs = load_pine_labs_dataset(all_pairs_fname)[:end_index]
    src_seqs = [seq for seq,tag in all_pairs]
    print(f'loaded {len(src_seqs)} sentences')

    model = gensim.models.Word2Vec(
        src_seqs,
        size=emb_size,
        window=span,
        min_count=1,
        workers=n_cpus)
    print(f'prepared the word2vec model {model}, about to train')
    t0 = time()
    model.train(src_seqs, total_examples=len(src_seqs), epochs=epochs)
    t = round(time()-t0,2)
    print(f'trained w2v model in {t} sec.')

    if save_gensim_model:
        model.save(gensim_model_fname)
        print(f'word2vec trained model, saved in file: {gensim_model_fname}')

    vocab, vecs = get_vocab_embeddings(model)
    print(f'trained on a vocab of {len(vocab)} words and obtained embedding matrix of shape: {vecs.shape}')

    word_id_map = {w:i for i,w in enumerate(vocab)}
    with open(word_id_map_fname,'w') as fh:
        json.dump(word_id_map,fh)

    np.savetxt(word_vectors_fname,X=vecs)

    print(f'saved word id map in {word_id_map_fname} and embeddings in {word_vectors_fname}')


if __name__ == '__main__':
    main(span=5, epochs=20, save_gensim_model=False)



