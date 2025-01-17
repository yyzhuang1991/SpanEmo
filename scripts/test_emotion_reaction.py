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
    strs.append(f"  emotion: p = {pres[1]}, r = {recs[1]}, f1 = {f1s[1]}\n  non-emotional: p = {pres[0]}, r = {recs[0]}, f1 = {f1s[0]}\nmacro pre: {macro_pre}, recl: {macro_rec}, f1: {macro_f1}\n")
    

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

test_dataset_with_prev = PredictDataClass(int(args['--max-length']), args['--test-path'], include_prev_sentence = True)
test_data_loader_with_prev = DataLoader(test_dataset_with_prev,
                              batch_size=int(args['--test-batch-size']),
                              shuffle=False)

# test_dataset = PredictDataClass(int(args['--max-length']), args['--test-path'], include_prev_sentence = 0)
# test_data_loader = DataLoader(test_dataset,
#                               batch_size=int(args['--test-batch-size']),
#                               shuffle=False)
# apply predictions on unlabeled data 
model = SpanEmo(lang=args['--lang'])
true_labels = [0] * len(test_dataset_with_prev)
learn = Predictor(model, test_data_loader_with_prev, model_path='models/' + args['--model-path'])
pred = learn.predict(device=device)

def save_label(pred_labels, outfile):
    pred = []
    print(f"saving predictions to {outfile}")
    for pred_label in pred_labels:
        if len(pred_label):
            pred.append(1)
        else:
            pred.append(0)
    with open(outfile, "w") as f:
        json.dump(pred, f)    

save_label(pred, "my_test_data/predictions.json")
# with open(args['--test-path'] + f".out.{name}.json", "w") as f:
#     json.dump(pred, f)
cal_score(true_labels, pred, outfile = args['--test-path']+"z.eval")

# for i in range(3,8):
#     print(f" window = {i}")
#     test_dataset_5words = PredictDataClass(int(args['--max-length']), args['--test-path'], include_prev_sentence = 0, kwords = i)
#     test_data_loader_5words = DataLoader(test_dataset_5words,
#                                   batch_size=int(args['--test-batch-size']),
#                                   shuffle=False)

#     true_labels = test_dataset_5words.labels
#     learn = Predictor(model, test_data_loader_5words, model_path='models/' + args['--model-path'])
#     pred = learn.predict(device=device)
#     # with open(args['--test-path'] + f".out.{name}.json", "w") as f:
#     #     json.dump(pred, f)
#     cal_score(true_labels, pred, outfile = args['--test-path']+"z.eval")



    # test_dataset_3words = PredictDataClass(int(args['--max-length']), args['--test-path'], include_prev_sentence = 0, kwords = 3)
    # test_data_loader_3words = DataLoader(test_dataset_3words,
    #                               batch_size=int(args['--test-batch-size']),
    #                               shuffle=False)


    # true_labels = test_dataset_5words.labels

    #############################################################################
    # Run the model on a Test set
    #############################################################################

    # for loader, name in zip([test_data_loader, test_data_loader_with_prev, test_data_loader_5words, test_data_loader_3words], ["current-sent", "with-prev", "5words", "3words"]):
    #     print(f"-----{name}-----")
    #     model = SpanEmo(lang=args['--lang'])

    #     learn = Predictor(model, loader, model_path='models/' + args['--model-path'])
    #     pred = learn.predict(device=device)
    #     with open(args['--test-path'] + f".out.{name}.json", "w") as f:
    #         json.dump(pred, f)

    #     cal_score(true_labels, pred, outfile = args['--test-path']+f"{name}.eval")



