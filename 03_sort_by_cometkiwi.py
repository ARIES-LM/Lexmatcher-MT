
import sys
import numpy as np

src_file = sys.argv[1]
tgt_file = sys.argv[2]
comet_file = sys.argv[3]

# negative
comet_scores = [float(l.strip()) for l in open(comet_file)]

srclens = [len(l.split()) for l in open(src_file)]

srclens = np.array(srclens)
srclens = np.exp(1/srclens)

comet_scores = np.array(comet_scores)

comet_scores_norm = comet_scores / srclens

print("ori median {}, after length norm {}".format(np.median(comet_scores), np.median(comet_scores_norm)))
print("ori min {}, after length norm {}".format(np.min(comet_scores), np.min(comet_scores_norm)))

idx = np.argsort(comet_scores_norm * -1.0)

src = open(src_file).readlines()
tgt = open(tgt_file).readlines()

thres = sys.argv[4]
thres = float(thres)

print('threshold, ', thres)

'''
for i in idx:
    if comet_scores_norm[i] < thres:
'''

import random 

with open(src_file+'.kiwi', 'w') as fwsrc, open(tgt_file+'.kiwi', 'w') as fwtgt:
    for i in idx:
        if comet_scores_norm[i] > thres:
            #print(src[i].strip(), '||', tgt[i].strip())
            print(src[i].strip(), file=fwsrc)
            print(tgt[i].strip(), file=fwtgt)
        else:
            # examples
            if random.random() < 0.3 and comet_scores_norm[i] > 0.7:
                print(comet_scores_norm[i], '|', src[i].strip(), '|', tgt[i].strip())


