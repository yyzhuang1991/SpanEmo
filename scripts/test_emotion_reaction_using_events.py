
from learner import Predictor
from model import SpanEmo
from data_loader import PredictDataClass
from torch.utils.data import DataLoader
import torch
from docopt import docopt
import numpy as np
import json 
from sklearn.metrics import precision_recall_fscore_support



args = docopt(__doc__)
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

test_dataset = PredictDataClass(128, args['--test-path'], include_prev_sentence = False, use_events = True)
test_data_loader = DataLoader(test_dataset,
                              batch_size=32,
                              shuffle=False)
model = SpanEmo(lang="English")

learn = Predictor(model, test_data_loader, model_path='models/' + args['--model-path'])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
pred = learn.predict(device=device)

ephrase2emotion = {}
for ephrase, p in zip(test_dataset.ephrases, pred):
    ephrase2emotion[ephrase] = 'emotional' if len(p) else 'neutral' 

# collect 
data = test_dataset.input_data 
# map example id 2 ephrase  
exid2ephrase = {} 
for ephrase in test_dataset.ephrase2exid:
    for exid in test_dataset.ephrase2exid[ephrase]:
        if exid not in exid2ephrase:
            exid2ephrase[exid] = [] 
        exid2ephrase[exid].append(ephrase)

# collect affective score for all events that have a body part  
for exid in range(len(data)):
    
    emotion_label = 'neutral'
    ephrase_pred = [] #(phrase, affective label)

    for ephrase in exid2ephrase.get(exid, []):
        ephrase_pred.append((ephrase, ephrase2emotion[ephrase]))
        if ephrase2emotion[ephrase] == 'emotional':
            emotion_label = 'emotional'

    d = data[exid]
    d['ephrase_pred'] = ephrase_pred
    d['emotion_label'] = emotion_label

label_mapping = {"1":0, "2":1, "3":0} 
true = [label_mapping[d['label'][0]] for d in data]

pred = [0 if d['emotion_label'] == 'neutral' else 1 for d in data]

pres, recs, f1s, _  = precision_recall_fscore_support(true, pred, average = None)
macro_pre, macro_rec, macro_f1, _ = precision_recall_fscore_support(true, pred, average = 'macro')
micro_pre, micro_rec, micro_f1, _ = precision_recall_fscore_support(true, pred, average = 'micro')


print(f"  emotion: p = {pres[1]}, r = {recs[1]}\n  non-emotional: p = {pres[0]}, r = {recs[0]}\nmacro f1: {macro_f1}\n")