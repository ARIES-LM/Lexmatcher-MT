import json
import random
from nltk.tokenize import word_tokenize
import sys
import string
from nltk.corpus import stopwords

stopwords = stopwords.words('english') + list(string.punctuation)
stopwords = set(stopwords)

from nltk.corpus import wordnet as wn


wn_lang_codes = {'zh':'cmn', 'de':'deu', 'ru':"rus", "is":"isl", "cs":"ces"}

def get_words_by_language(tgt_lang):

    language_code = wn_lang_codes[tgt_lang]
    print('language_code', language_code)

    exown_lang_code = f"{language_code}_wikt"

    from collections import defaultdict
    bidict = defaultdict(list)

    for synset in wn.all_synsets():
        deninition = synset.definition()

        for lemma in synset.lemmas():
            en_lemma = lemma.name()

            tgt = synset.lemmas(exown_lang_code)

            try:
                tgt_2 = synset.lemmas(language_code)
            except:
                tgt_2 = []

            tgt_all = tgt+tgt_2

            if len(tgt_all) == 0:
                continue

            for tgt_i in tgt_all:
                tgt_word = tgt_i.name()

                if '+' in tgt_word:
                    tgt_word = tgt_word.split('+')[0]

                bidict[en_lemma].append(tgt_word)
    
    sampledkeys = list(bidict.keys())

    with open(f'wn_bidict_en{tgt_lang}.json', 'w') as fw:
        for k in sampledkeys:
            v = bidict[k]
            v = list(set(v))
            jsoned = json.dumps({k:v}, ensure_ascii=False)
            print(jsoned, file=fw)

if __name__ == '__main__':
    wn.add_exomw()

    tgts = ['de', 'ru', 'zh']
    for tgt in tgts:
        get_words_by_language(tgt)
