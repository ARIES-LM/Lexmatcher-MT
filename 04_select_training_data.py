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

from fuzzywuzzy import fuzz


stop = set(stopwords.words('english') + list(string.punctuation))
numeric = set(list(string.digits))

random.seed(42)
np.random.seed(42)

def random_selection(src_file, tgt_file, num, min_token=25):
    src_content = open(src_file).readlines()
    tgt_content = open(tgt_file).readlines()

    idx = list(range(len(src_content)))
    random.shuffle(idx)

    print('total', len(idx))

    with open(f'train.rd{num}.src', 'w') as f1, open(f'train.rd{num}.tgt', 'w') as f2:
        for i in idx:
            if num == 0:
                break

            if len(src_content[i].split()) > min_token:
                print(src_content[i].strip(), file=f1)
                print(tgt_content[i].strip(), file=f2)

                num -= 1

def find_match(sentence, kw, mode='fuzzy', threshold=90):
    '''
    simple match of exact keyword in sentence
    :param sentence: str
    :param kw: str
    :param mode: str ('regex', 'fuzzy')
    :param threshold: int
    :return: match: 0 if no match, 1 if match
    '''
    if mode == 'regex':
        #regex match
        try:
            regex_match = re.search(kw, sentence)
        except:
            regex_match = re.search(re.escape(kw), re.escape(sentence))

        if regex_match is not None:
            match = 1
        else:
            match = 0

    # fuzzy match
    elif mode == 'fuzzy':
        fuzzy_match = fuzz.partial_ratio(kw, sentence)
        fuzzy_type = 'fuzzy'+str(threshold)
        if len(sentence) > 200:
            # adjustment made for long sequences: if sentence is longer than approx. 200 chars, the fuzzy score
            # starts sharply degrading
            fuzzy_match = fuzzy_match * 2
        if fuzzy_match >= threshold:
            match = 1
        else:
            match = 0

    return match


def read_align(alignfile):
    s = {}

    for l in open(alignfile):
        # if 'json' in alignfile:
        l = json.loads(l)

        key, v = l.popitem()
        key = key.lower()

        v = [vi.lower() for vi in v]
        v = list(set(v))

        s[key] = v

    print('len align', len(s))
    return s


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


def _match_step(args, w_lem, align_freq, tmp_align_freq, nshot, aligns_meet, lem_tokens_tgt_str, lem_tokens_tgt_set):
    if w_lem in stop or w_lem in numeric:
        return

    src = args.source_lang
    tgt = args.target_lang

    if w_lem in align_freq:
        for wtgt in align_freq[w_lem]:
            # r1
            if align_freq[w_lem][wtgt] >= nshot:
                continue
            # r2
            # if align_freq[w_lem][wtgt] + tmp_align_freq[w_lem][wtgt] >= nshot:
            #     continue

            if '_' in wtgt:
                wtgt_nwords = wtgt.replace('_', ' ')

                if args.fuzzy:
                    if tgt == 'zh':
                        sent = lem_tokens_tgt_str.replace(" ", "")
                    else:
                        sent = lem_tokens_tgt_str

                    if find_match(sent, wtgt_nwords, mode='fuzzy', threshold=90):
                        align_freq[w_lem][wtgt] += 1
                        aligns_meet.append(f'{w_lem}|{wtgt}')
                else:
                    if wtgt_nwords in lem_tokens_tgt_str:
                        align_freq[w_lem][wtgt] += 1
                        aligns_meet.append(f'{w_lem}|{wtgt}')

            else:
                wtgt_nwords = wtgt

                if args.fuzzy:
                    if tgt == 'zh':
                        sent = lem_tokens_tgt_str.replace(" ", "")
                    else:
                        sent = lem_tokens_tgt_str

                    if find_match(sent, wtgt_nwords, mode='fuzzy', threshold=90):
                        align_freq[w_lem][wtgt] += 1
                        aligns_meet.append(f'{w_lem}|{wtgt}')
                else:
                    if wtgt_nwords in lem_tokens_tgt_set:
                        align_freq[w_lem][wtgt] += 1
                        aligns_meet.append(f'{w_lem}|{wtgt}')

def select_with_align_with_prelemma(args):
    alignment = read_align(args.align_path)

    from collections import defaultdict

    align_freq = {}
    for k in alignment:
        align_freq[k] = {v: 0 for v in alignment[k]}

    nshot = args.nshot
    minlen = 15
    maxlen = 80

    with open(f'{args.src_path}.lem') as frsrclem, open(f'{args.tgt_path}.lem') as frtgtlem, \
        open(f'{args.src_path}.{args.align_type}.nshot{nshot}minali{args.min_align_in_sent}', 'w') as fwsrc, \
            open(f'{args.tgt_path}.{args.align_type}.nshot{nshot}minali{args.min_align_in_sent}', 'w') as fwtgt, \
            open(f'{args.src_path}.{args.align_type}.nshot{nshot}minali{args.min_align_in_sent}.align', 'w') as fwali:

        for i, (lsrc, ltgt, lsrclem, ltgtlem) in enumerate(zip(open(args.src_path), open(args.tgt_path), frsrclem, frtgtlem)):

            if i % 1000000 == 0:
                print(i)

            lsrclem, ltgtlem = lsrclem.lower(), ltgtlem.lower()

            tmp_align_freq = defaultdict(lambda: defaultdict(int))

            if 1:
                lem_tokens_src = lsrclem.strip().split()
                lem_tokens_tgt_str = ltgtlem
                lem_tokens_tgt_set = set(ltgtlem.strip().split())

                if len(lem_tokens_src) > maxlen or len(lem_tokens_src) < minlen:
                    continue

                aligns_meet = []

                for w_lem in set(lem_tokens_src):
                    _match_step(args, w_lem, align_freq, tmp_align_freq, nshot, aligns_meet,
                                lem_tokens_tgt_str, lem_tokens_tgt_set)

                for bgsrc in set(nltk.bigrams(lem_tokens_src)):
                    bgsrc = '_'.join(bgsrc)
                    _match_step(args, bgsrc, align_freq, tmp_align_freq, nshot, aligns_meet,
                                lem_tokens_tgt_str, lem_tokens_tgt_set)

                if len(set(aligns_meet)) >= args.min_align_in_sent:
                    # r2
                    # for tmpsrc, tmpitem in tmp_align_freq.items():
                    #     for tmptgt, tmpfreq in tmpitem.items():
                    #         align_freq[tmpsrc][tmptgt] += tmpfreq

                    print(lsrc.strip(), file=fwsrc)
                    print(ltgt.strip(), file=fwtgt)
                    print('\t'.join(aligns_meet), file=fwali)


    allpairs = 0
    matchpairs = 0.0

    with open(args.write_freq_path, 'w') as fw:
        for w in align_freq:
            print(w, align_freq[w])

            for wtgt, wfreq in align_freq[w].items():
                allpairs += 1
                if wfreq != 0:
                    matchpairs += 1

            ordered = sorted(align_freq[w].items(), key=lambda x: x[1], reverse=True)
            jsoned = json.dumps({w: ordered}, ensure_ascii=False)
            print(jsoned, file=fw)

    print(f'min_align_in_sent {args.min_align_in_sent}, match rate {matchpairs / allpairs}')



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='compute statistic for quality filtering')
    parser.add_argument('--src_path', type=str, help='path to jsonl dataset')
    parser.add_argument('--tgt_path', type=str, help='path to jsonl dataset')

    parser.add_argument('--align_path', type=str, help='path to jsonl dataset')

    parser.add_argument('--token', type=int, default=0, help='0: not use tokenizer, 1: use tokenizer')
    parser.add_argument('--min_align_in_sent', type=int, default=1)

    parser.add_argument('--target_lang', type=str)
    parser.add_argument('--source_lang', type=str)

    parser.add_argument('--nshot', type=int, default=3)

    parser.add_argument('--write_freq_path', type=str, help='write freq')

    parser.add_argument('--fuzzy', type=int, default=0, help='fuzzy matching or exact matching')

    args = parser.parse_args()

    print('lexmatcher selection')
    select_with_align_with_prelemma(args)


