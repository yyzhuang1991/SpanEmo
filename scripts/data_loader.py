from torch.utils.data import Dataset
from transformers import BertTokenizer, AutoTokenizer
from tqdm import tqdm
import torch
import pandas as pd
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.classes.preprocessor import TextPreProcessor
import json 

import sys 
from os.path import abspath, dirname, join
tuple_util_path = join(dirname(dirname(dirname(abspath(__file__)))), "emotion-reaction/model-scripts")
print(tuple_util_path)
sys.path.append(tuple_util_path)
import tuple_utils

# classes = ['1','2a','2b','3a', '3b'] 
classes = {
    "1": 0, 
    "3":0, 
    "3a":0, 
    "3b":0,
    "2":1, 
    "2a": 1, 
    "2b":1, 
    }


lemma_should_get_word = set([
    "i",
    "my",
    "me",
    "we",
    "our",
    "us",
    "you",
    "your",
    "he",
    "him",
    "his",
    "she",
    "her",
    "hers",
    "they",
    "them",
    "their"
    ])

def get_lemmatized_ephrase_with_correct_pronoun(comp_idx, neg_idx, lemmas, words):
    ret = [] 
    for compid, nid in zip(comp_idx, neg_idx):
        if nid is not None: 
            ret.append('not')
        for cid in compid:
            l = lemmas[cid].lower()
            w = words[cid].lower()
            if l in lemma_should_get_word:
                ret.append(w)
            else:
                ret.append(l)
    return " ".join(" ".join(ret).split())

def get_tuples_with_bp(event_dicts, bp_idx, lemmas, words):
    event_phrases = [] 
    for edict in event_dicts:
        word_idx = [ind for comp_ind in edict['event_inds'] for ind in comp_ind]
        if bp_idx in word_idx:
            ephrase = get_lemmatized_ephrase_with_correct_pronoun(edict['event_inds'], edict['neg_inds'], lemmas, words)
            event_phrases.append(ephrase)

    return list(set(event_phrases))


def twitter_preprocessor():
    preprocessor = TextPreProcessor(
        normalize=['url', 'email', 'phone', 'user'],
        annotate={"hashtag", "elongated", "allcaps", "repeated", 'emphasis', 'censored'},
        all_caps_tag="wrap",
        fix_text=False,
        segmenter="twitter_2018",
        corrector="twitter_2018",
        unpack_hashtags=True,
        unpack_contractions=True,
        spell_correct_elong=False,
        tokenizer=SocialTokenizer(lowercase=True).tokenize).pre_process_doc
    return preprocessor


class DataClass(Dataset):
    def __init__(self, args, filename):
        self.args = args
        self.filename = filename
        self.max_length = int(args['--max-length'])
        self.data, self.labels = self.load_dataset()

        if args['--lang'] == 'English':
            self.bert_tokeniser = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        elif args['--lang'] == 'Arabic':
            self.bert_tokeniser = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")
        elif args['--lang'] == 'Spanish':
            self.bert_tokeniser = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")

        self.inputs, self.lengths, self.label_indices = self.process_data()

    def load_dataset(self):
        """
        :return: dataset after being preprocessed and tokenised
        """
        df = pd.read_csv(self.filename, sep='\t')
        x_train, y_train = df.Tweet.values, df.iloc[:, 2:].values
        print(x_train[0], y_train[0])
        return x_train, y_train

    def process_data(self):
        desc = "PreProcessing dataset {}...".format('')
        preprocessor = twitter_preprocessor()

        if self.args['--lang'] == 'English':
            segment_a = "anger anticipation disgust fear joy love optimism hopeless sadness surprise or trust?"
            label_names = ["anger", "anticipation", "disgust", "fear", "joy",
                           "love", "optimism", "hopeless", "sadness", "surprise", "trust"]
        elif self.args['--lang'] == 'Arabic':
            segment_a = "غضب توقع قرف خوف سعادة حب تفأول اليأس حزن اندهاش أو ثقة؟"
            label_names = ['غضب', 'توقع', 'قر', 'خوف', 'سعادة', 'حب', 'تف', 'الياس', 'حزن', 'اند', 'ثقة']

        elif self.args['--lang'] == 'Spanish':
            segment_a = "ira anticipaciÃ³n asco miedo alegrÃ­a amor optimismo pesimismo tristeza sorpresa or confianza?"
            label_names = ['ira', 'anticip', 'asco', 'miedo', 'alegr', 'amor', 'optimismo',
                           'pesim', 'tristeza', 'sorpresa', 'confianza']

        inputs, lengths, label_indices = [], [], []
        for x in tqdm(self.data, desc=desc):
            x = ' '.join(preprocessor(x))
            x = self.bert_tokeniser.encode_plus(segment_a,
                                                x,
                                                add_special_tokens=True,
                                                max_length=self.max_length,
                                                pad_to_max_length=True,
                                                truncation=True)
            input_id = x['input_ids']
            input_length = len([i for i in x['attention_mask'] if i == 1])
            inputs.append(input_id)
            lengths.append(input_length)

            #label indices
            label_idxs = [self.bert_tokeniser.convert_ids_to_tokens(input_id).index(label_names[idx])
                             for idx, _ in enumerate(label_names)]
            label_indices.append(label_idxs)

        inputs = torch.tensor(inputs, dtype=torch.long)
        data_length = torch.tensor(lengths, dtype=torch.long)
        label_indices = torch.tensor(label_indices, dtype=torch.long)
        return inputs, data_length, label_indices

    def __getitem__(self, index):
        inputs = self.inputs[index]
        labels = self.labels[index]
        label_idxs = self.label_indices[index]
        length = self.lengths[index]
        return inputs, labels, length, label_idxs

    def __len__(self):
        return len(self.inputs)

def strip(text):
    try:
        return text.strip()
    except AttributeError:
        return text


class PredictDataClass(Dataset):
    def __init__(self, max_length, filename, include_prev_sentence, kwords = 0, use_events = 0):

        self.filename = filename
        self.max_length = max_length
        self.include_prev_sentence = include_prev_sentence
        self.kwords = kwords
        self.use_events = use_events
        
        assert ((self.kwords > 0 )+ self.include_prev_sentence) + self.use_events <= 1
        self.data, self.labels = self.load_dataset()
        print(f"{len(self.data)} inputs")
        self.bert_tokeniser = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.inputs, self.lengths, self.label_indices = self.process_data()
        


    def load_dataset(self):
        """
        :return: dataset after being preprocessed and tokenised
        """
        # if self.filename.endswith(".csv"):
        #     assert False # because kwords and prev context are not implemented for csv file
        #     fobj = pd.read_csv(self.filename, dtype = 'object', skipinitialspace=True)
        #     fobj = fobj[fobj['label'].notna()]
        #     prev_sentences = [t for t in fobj['prev sentence']]
        #     sentences = [t for t in fobj['sentence']]
        #     if 'label' in fobj:
        #         y_train = [classes[k] for k in fobj["label"]]
        #     else:
        #         y_train = [-1] * len(sentences)
            
        if self.filename.endswith(".json"):
            with open(self.filename) as f:
                fobj = json.load(f)
        else:
            raise NameError("File must end with .json")


        prev_sentences = [" ".join(t['prev_sentence']) for t in fobj]
        sentences = [t['sentence'] for t in fobj]
        
        if self.include_prev_sentence:
            x_train = [f"{p} {s}" for p, s in zip(prev_sentences, sentences)]
            if 'label' in fobj[0]:
                y_train = [classes[t['label']] for t in fobj]
            else:
                y_train = [-1] * len(sentences)

        elif self.kwords > 0: 

            sentences = []
            for t in fobj: 
                center = t['word_bound'][0] + 1
                words = t['sentence'].split() 
                left = max(0, center - self.kwords)
                right = min(len(words), center + self.kwords + 1)
                sentences.append(" ".join(words[left:right]))  

            # sentences = []
            # for t in fobj: 
            #     center = t['word_bound'][0] + 1
            #     words = t['sentence'].split() 
            #     left = max(0, center - kwords)
            #     right = min(len(words), center + kwords + 1)
            #     sentences.append(" ".join(words[left:right]))  
            x_train = sentences
            if 'label' in fobj[0]:
                y_train = [classes[t['label']] for t in fobj]
            else:
                y_train = [-1] * len(sentences)
        
        elif self.use_events:
            ephrase2exid = {} # ephrase to index in data
            data = fobj 
            for i, d in enumerate(data):
                d['mod_head'] = {int(mod): d['mod_head'][mod] for mod in d['mod_head']}
                event_dicts = tuple_utils.get_events_for_sentence(d['sentence'].split(), d['pos'], d['lemma'], d['mod_head'], d['ner'])
                bp_idx = d['word_bound'][0] + 1 
                ephrases = get_tuples_with_bp(event_dicts, bp_idx, d['lemma'], d['sentence'].split())
                for ephrase in ephrases:
                    if ephrase not in ephrase2exid:
                        ephrase2exid[ephrase] = [] 
                    ephrase2exid[ephrase].append(i)

            self.ephrase2exid = ephrase2exid
            self.input_data = data
            self.ephrases = list(ephrase2exid.keys())
            x_train = ephrases
            y_train = [-1] * len(sentences)
            

        return x_train, y_train

        
    def process_data(self):
        desc = "PreProcessing dataset {}...".format('')
        preprocessor = twitter_preprocessor()

        segment_a = "anger anticipation disgust fear joy love optimism hopeless sadness surprise or trust?"
        label_names = ["anger", "anticipation", "disgust", "fear", "joy",
                       "love", "optimism", "hopeless", "sadness", "surprise", "trust"]
   
        inputs, lengths, label_indices = [], [], []
        for x in tqdm(self.data, desc=desc):
            x = ' '.join(preprocessor(x))
            x = self.bert_tokeniser.encode_plus(segment_a,
                                                x,
                                                add_special_tokens=True,
                                                max_length=self.max_length,
                                                pad_to_max_length=True,
                                                truncation=True)
            input_id = x['input_ids']
            input_length = len([i for i in x['attention_mask'] if i == 1])
            inputs.append(input_id)
            lengths.append(input_length)

            #label indices
            label_idxs = [self.bert_tokeniser.convert_ids_to_tokens(input_id).index(label_names[idx])
                             for idx, _ in enumerate(label_names)]
            label_indices.append(label_idxs)

        inputs = torch.tensor(inputs, dtype=torch.long)
        data_length = torch.tensor(lengths, dtype=torch.long)
        label_indices = torch.tensor(label_indices, dtype=torch.long)
        return inputs, data_length, label_indices

    def __getitem__(self, index):
        inputs = self.inputs[index]
        labels = self.labels[index]
        label_idxs = self.label_indices[index]
        length = self.lengths[index]
        return inputs, labels, length, label_idxs

    def __len__(self):
        return len(self.inputs)
