import torch
from torchtext import data
from torchtext.vocab import Vectors
#import spacy
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.train_iterator = None
        self.test_iterator = None
        self.val_iterator = None
        self.vocab = []
        self.word_embeddings = {}
    
    def parse_label(self, label):
        '''
        Get the actual labels from label string
        Input:
            label (string) : labels of the form 'positive,negative,neutral'
        Returns:
            label (int) : integer value corresponding to label string
        '''
        sentiment = ['positive','negative','neutral']
        return sentiment.index(label.strip())+1 

    def get_pandas_df(self, filename):
        '''
        Load the data into Pandas.DataFrame object
        This will be used to convert data to torchtext object
        '''
        l = 0
        print('start:',filename)
        with open(filename, 'r') as datafile:
            #data = [line.strip().split(',', maxsplit=1) for line in datafile]
            data = [line.strip().split('\t', maxsplit=1) for line in datafile]
            #print(data)
            #print('read completed')
            data_text = list(map(lambda x: x[0], data))
            #print('ok')
            data_label = list(map(lambda x: self.parse_label(x[1]), data))
            #print(data_text,data_label)
        full_df = pd.DataFrame({"text":data_text, "label":data_label})
        #print('End:',filename)
        return full_df

    
    def load_data(self, train_file, w2v_file=None, test_file=None, val_file=None):
        '''
        Loads the data from files
        Sets up iterators for training, validation and test data
        Also create vocabulary and word embeddings based on the data
        
        Inputs:
            w2v_file (String): absolute path to file containing word embeddings (GloVe/Word2Vec)
            train_file (String): absolute path to training file
            test_file (String): absolute path to test file
            val_file (String): absolute path to validation file
        '''
        '''For formal English tokenizer'''
        #NLP = spacy.load('en')
        #tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]
        '''For informal code-mix tokenizer'''
        tokenizer = lambda sent: [x for x in sent.split() if x != '']
        
        # Creating Field for data
        TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=self.config.max_sen_len)
        LABEL = data.Field(sequential=False, use_vocab=False)
        datafields = [("text",TEXT),("label",LABEL)]
        
        # Load data from pd.DataFrame into torchtext.data.Dataset
        train_df = self.get_pandas_df(train_file)
        train_examples = [data.Example.fromlist(i, datafields) for i in train_df.values.tolist()]
        train_data = data.Dataset(train_examples, datafields)

        if test_file:
            test_df = self.get_pandas_df(test_file)
            test_examples = [data.Example.fromlist(i, datafields) for i in test_df.values.tolist()]
            test_data = data.Dataset(test_examples, datafields)
        
        # If validation file exists, load it. Otherwise get validation data from training data
        if val_file:
            val_df = self.get_pandas_df(val_file)
            val_examples = [data.Example.fromlist(i, datafields) for i in val_df.values.tolist()]
            val_data = data.Dataset(val_examples, datafields)
        else:
            train_data, val_data = train_data.split(split_ratio=0.8)
        
        '''Pre-trained Embeddings'''
        if w2v_file:
            TEXT.build_vocab(train_data, vectors=Vectors(w2v_file))
        self.word_embeddings = TEXT.vocab.vectors
        self.vocab = TEXT.vocab
        '''Without pre-trained Embeddings'''
        #TEXT.build_vocab(train_data)
        #print('vocab:',len(TEXT.vocab))
        #self.word_embeddings = TEXT.vocab.stoi
        #self.vocab = TEXT.vocab

        self.train_iterator = data.BucketIterator(
            (train_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=True)
        
        self.val_iterator = data.BucketIterator(
            (val_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=False)

        # self.val_iterator, self.test_iterator = data.BucketIterator.splits(
        #     (val_data, test_data),
        #     batch_size=self.config.batch_size,
        #     sort_key=lambda x: len(x.text),
        #     repeat=False,
        #     shuffle=False)
        
        print ("Loaded {} training examples".format(len(train_data)))
        #print ("Loaded {} test examples".format(len(test_data)))
        print ("Loaded {} validation examples".format(len(val_data)))


def evaluate_model(model, iterator):
    all_preds = []
    all_y = []
    for idx,batch in enumerate(iterator):
        if torch.cuda.is_available():
            x = batch.text.cuda()
        else:
            x = batch.text
        y_pred = model(x)
        predicted = torch.max(y_pred.cpu().data, 1)[1] + 1
        all_preds.extend(predicted.numpy())
        all_y.extend(batch.label.numpy())
    '''Accuracy'''
    #score = accuracy_score(all_y, np.array(all_preds).flatten())
    '''F1-score'''
    score = f1_score(all_y, np.array(all_preds).flatten(),average='micro')
    return score
