import json
import numpy as np
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def gather_hyps(case, num_candidates=5):
    hyps = [case['ref']['content']]
    scores = [case['ref']['score']]
    for i in range(num_candidates):
        hyps.append(case[f'hyp_{i}']['content'])
        scores.append(case[f'hyp_{i}']['score'])
    return {
        'prompt': case['prompt'],
        'source_lang': case['source_lang'],
        'target_lang': case['target_lang'],
        'src': case['src'],
        'hyps': hyps,
        'scores':scores
    }

def sorted_by_delta(pd, reversed=False):
    return sorted(pd, key=lambda x: x['chosen_score'] - x['rejected_score'], reverse=reversed)

def sorted_candidates(data):

    sorted_indices = np.argsort(data['scores'])[::-1]
    candidates = [ data['hyps'][o] for o in sorted_indices]
    scores = [ data['scores'][o] for o in sorted_indices]

    return candidates, scores

def filter(data, winner_threshold = 0.7, loser_threshold = 0.3, gap_threshold = 0.01):
    candidates, scores = sorted_candidates(data)
    winners = [candidates[0]]
    losers = [candidates[-1]]

    pair_data = {
        "prompt": data['prompt'],
        "src": data['src'],
        'source_lang': data['source_lang'],
        'target_lang': data['target_lang'],
        'chosen': winners[0],
        'rejected': losers[0],
        "chosen_score": scores[0],
        "rejected_score": scores[-1],
    }
    if pair_data['chosen_score'] > winner_threshold and pair_data['rejected_score'] > loser_threshold and pair_data['chosen_score']- pair_data['rejected_score']>=gap_threshold:
        return [pair_data]


def dividedbydirection(preference_data):
    en_zh_preference_data = []
    zh_en_preference_data = []
    en_ru_preference_data = []
    ru_en_preference_data = []
    en_de_preference_data = []
    de_en_preference_data = []

    for d in preference_data:
        if d['target_lang'] == "Russian":
            en_ru_preference_data.append(d)
        elif d['target_lang'] == "Chinese":
            en_zh_preference_data.append(d)
        elif d['target_lang'] == "German":
            en_de_preference_data.append(d)
        elif d['target_lang'] == "English":
            if d['source_lang'] == "Russian":
                ru_en_preference_data.append(d)
            elif d['source_lang'] == "Chinese":
                zh_en_preference_data.append(d)
            elif d['source_lang'] == "German":
                de_en_preference_data.append(d)

    return (
        en_zh_preference_data,
        zh_en_preference_data,
        en_ru_preference_data,
        ru_en_preference_data,
        en_de_preference_data,
        de_en_preference_data,
    )

def extract_top_K(direction_data, k):
    return direction_data[:k]
def init_agrs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("-wt", "--winner_threshold", type=float)
    parser.add_argument("-lt", "--loser_threshold", type=float)
    parser.add_argument("-sd", "--score_difference", type=float)
    parser.add_argument("-k","--top_k", help="getting top k samples per direction" )
    parser.add_argument("--save_path", type=str, help="the path used to save hypothesis from model", required=True)
    return parser


def main():
    arg_parser = init_agrs()
    args = arg_parser.parse_args()
    data_path = args.data_path
    save_path = args.save_path
    winner_threshold = args.wt
    loser_threshold = args.lt
    score_difference = args.sd
    k = args.k
    div_data = read_json(data_path)
    div_data = list(map(gather_hyps, div_data))

    process = partial(filter, winner_threshold = winner_threshold, loser_threshold = loser_threshold, score_difference = score_difference)
    with ProcessPoolExecutor(max_workers=8) as executor:
        pd = []
        for result in tqdm(executor.map(process, div_data), total=len(div_data)):
            pd.extend(result)

    pd = sorted_by_delta(pd, reversed=True)
    all_langs = dividedbydirection(pd)
    all_data = []
    for dir_data in all_langs:
        all_data.extend(extract_top_K(dir_data, k))

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all, f, ensure_ascii=False, indent=4)



if __name__ == '__main__':
    main()
