import os

import json
import sys

from comet import download_model, load_from_checkpoint

model_path = "/home/yinyongjing/workspace/tfs/models/wmt22-cometkiwi-da/checkpoints/model.ckpt"
model = load_from_checkpoint(model_path)

'''
# Data must be in the following format:
data = [
    {
        "src": "10 到 15 分钟可以送到吗",
        "mt": "Can I receive my food in 10 to 15 minutes?",
    },
    {
        "src": "Pode ser entregue dentro de 10 a 15 minutos?",
        "mt": "Can you send it for 10 to 15 minutes?",
    }
]
# Call predict method:
model_output = model.predict(data, batch_size=8, gpus=1)
print(model_output)
print(model_output.scores) # sentence-level scores
print(model_output.system_score) # system-level score
exit()
'''
data=[]
with open(sys.argv[1]) as fsrc, open(sys.argv[2]) as fhyp:
    for src, hypo in zip(fsrc, fhyp):
        data.append(
             {
                 "src": src,
                 "mt": hypo,
             }
        )


ngpu = int(sys.argv[4])

model_output = model.predict(data, batch_size=512, gpus=ngpu)

scores = model_output.scores
scores = map(str, scores)

with open(sys.argv[3], 'w') as fw:
    fw.write('\n'.join(scores))
    fw.write('\n')

