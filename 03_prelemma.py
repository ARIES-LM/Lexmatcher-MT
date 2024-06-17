import random
from simplemma import text_lemmatizer

from pathlib import Path
import json
import numpy as np
import os
from itertools import islice
from joblib import Parallel, delayed
from nltk.tokenize import word_tokenize
from collections import Counter
from tqdm import tqdm
from multiprocessing import Pool

import string
from nltk.corpus import stopwords
import nltk
import jieba
import lxml.html
import spacy


stop = set(stopwords.words('english') + list(string.punctuation))
numeric = set(list(string.digits))

random.seed(42)
np.random.seed(42)

pipedisable = ["tokenizer", "parser", "ner"]

def get_nlp_tool(lang):
    if lang == 'en':
        print('load spacy')
        return spacy.load('en_core_web_lg')

    if lang == 'zh':
        print('not use spacy zh')
        return None
    else:
        try:
            print('load spacy')
            return spacy.load(f"{lang}_core_news_lg")
        except Exception:
            print('no module found in spacy')
            return None

def _process_select(args, src_tool, tgt_tool, batch_src, batch_tgt, n_process, tgt, fwsrc, fwtgt):
    if src_tool is not None:
        doc_src = list(src_tool.pipe(batch_src, n_process=n_process, disable=pipedisable))
    else:
        #if args.source_lang == "zh":
        #    doc_src = [jieba.cut(l.strip()) for l in batch_src]
        #else:
        doc_src = [l.strip().split() for l in batch_src]
    
    if tgt_tool:
        doc_tgt = list(tgt_tool.pipe(batch_tgt, n_process=n_process, disable=pipedisable))
    else:
        #if args.target_lang == "zh":
        #    doc_tgt = [jieba.cut(l.strip()) for l in batch_tgt]
        #else:
        doc_tgt = [l.split() for l in batch_tgt]

    for lsrc, ltgt in zip(doc_src, doc_tgt):
        if src_tool is None:
            lem_tokens_src = lsrc
        else:
            lem_tokens_src = [w.lemma_ for w in lsrc]
        
        if tgt_tool is None:
            lem_tokens_tgt = ltgt
        else:
            lem_tokens_tgt = [w.lemma_ for w in ltgt]

        print(' '.join(lem_tokens_src), file=fwsrc)
        print(' '.join(lem_tokens_tgt), file=fwtgt)


def lemma(args):
    bz = 16000
    n_process = 16

    batch_src = []
    batch_tgt = []
    
    src = args.source_lang
    src_tool = get_nlp_tool(src)

    tgt = args.target_lang
    tgt_tool = get_nlp_tool(args.target_lang)

    with open(f'{args.ds_path}.lem', 'w') as fwsrc, open(f'{args.tgt_path}.lem', 'w') as fwtgt:

        for i, (lsrc, ltgt) in enumerate(zip(open(args.ds_path), open(args.tgt_path))):
            if i % 100000 == 0:
                print(i)

            batch_src.append(lsrc.strip())
            batch_tgt.append(ltgt.strip())
            
            if len(batch_src) == bz:
                _process_select(args, src_tool, tgt_tool, batch_src, batch_tgt, n_process,
                                tgt, fwsrc, fwtgt)
                batch_tgt, batch_src = [], []

        if len(batch_src) > 0:
            _process_select(args, src_tool, tgt_tool, batch_src, batch_tgt, n_process,
                            tgt, fwsrc, fwtgt)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='compute statistic for quality filtering')
    parser.add_argument('--ds_path', type=str, help='path to jsonl dataset')
    parser.add_argument('--tgt_path', type=str, help='path to jsonl dataset')

    parser.add_argument('--token', type=int, default=0, help='0: not use tokenizer, 1: use tokenizer')
    parser.add_argument('--target_lang', type=str)
    parser.add_argument('--source_lang', type=str)


    args = parser.parse_args()

    lemma(args)
    print('done')


