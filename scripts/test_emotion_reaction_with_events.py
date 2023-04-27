"""
Usage:
    main.py [options]

Options:
    -h --help                         show this screen
    --model-path=<str>                path of the trained model
    --max-length=<int>                text length [default: 128]
    --seed=<int>                      seed [default: 0]
    --test-batch-size=<int>           batch size [default: 32]
    --lang=<str>                      language choice [default: English]
    --test-path=<str>                 file path of the test set [default: ]
"""
from learner import Predictor
from model import SpanEmo
from data_loader import PredictDataClass
from torch.utils.data import DataLoader
import torch
from docopt import docopt
import numpy as np
import json 
from sklearn.metrics import precision_recall_fscore_support


def cal_score(true_labels, pred_labels, outfile):
    true = true_labels 
    pred = []
    for true_label, pred_label in zip(true_labels, pred_labels):
        if len(pred_label):
            pred.append(1)
        else:
            pred.append(0)     

    labels = ['non-emotional', 'emotional']

    pres, recs, f1s, _  = precision_recall_fscore_support(true, pred, average = None)
    macro_pre, macro_rec, macro_f1, _ = precision_recall_fscore_support(true, pred, average = 'macro')
    micro_pre, micro_rec, micro_f1, _ = precision_recall_fscore_support(true, pred, average = 'micro')
    strs = []
    strs.append(f"NUM Samples: {len(true)}, NUM Emotional: {len([k for k in true if k == 1])}, Num Non-Emotional: {len([k for k in true if k == 0])}")
    strs.append(f"Precisions: {pres}")
    strs.append(f"Recalls: {recs}")
    strs.append(f"F1s: {f1s}")
    strs.append(f"Micro F1: {micro_f1}， Macro F1: {macro_f1}")

    strs = "\n".join(strs)
    print(strs)
    with open(outfile, 'w') as f:
        f.write(strs)

args = docopt(__doc__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if str(device) == 'cuda:0':
    print("Currently using GPU: {}".format(device))
    np.random.seed(int(args['--seed']))
    torch.cuda.manual_seed_all(int(args['--seed']))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    print("Currently using CPU")
#####################################################################
# Define Dataloaders
#####################################################################


test_dataset = PredictDataClass(int(args['--max-length']), args['--test-path'], use_events = True)
test_data_loader = DataLoader(test_dataset,
                              batch_size=int(args['--test-batch-size']),
                              shuffle=False)
model = SpanEmo(lang=args['--lang'])

learn = Predictor(model, test_data_loader, model_path='models/' + args['--model-path'])
pred = learn.predict(device=device)
# with open(args['--test-path'] + f".out.{name}.json", "w") as f:
#     json.dump(pred, f)
cal_score(true_labels, pred, outfile = args['--test-path']+f"{name}.eval")