# train.py

from utils import *
from model import *
from config import Config
import numpy as np
import sys
import torch.optim as optim
from torch import nn
import torch
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Fasttext SEMEval model')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epoch', type=int, default=10, help='maximum epoch number')
    parser.add_argument('--max_sen_len', type=int, default=20, help='maximum length')
    parser.add_argument('--lr', type=float, default=0.25, help='initial learning rate')
    parser.add_argument('--hidden_size', type=int, default=10, help='dropout ratio')
    parser.add_argument('--embed_size', type=int, default=50, help='embed size')
    parser.add_argument('--track', type=str, default='track.log', help='track file')
    parser.add_argument('--train', type=str, default='../data/train.tsv', help='training file')
    parser.add_argument('--valid', type=str, default='../data/valid.tsv', help='validation file')
    parser.add_argument('--test', type=str, default='../data/test.tsv', help='test file')
    parser.add_argument('--embedding', type=int, default=50, help='embed size')
    
    args = parser.parse_args()
    
    config = Config()
    config.lr = args.lr
    config.batch_size = args.batch_size
    config.max_epochs = args.epoch
    config.max_sen_len = args.max_sen_len 
    config.hidden_size = args.hidden_size
    config.embed_size = args.embed_size
    track_file = args.track
    train_file = args.train
    val_file = args.valid
    test_file = args.test

    w2v_file = args.embedding #'../data/skipgram.50d.txt'
    
    #Track file
    W = open(track_file,'w')
    
    dataset = Dataset(config)
    dataset.load_data( train_file=train_file, val_file=val_file, w2v_file=w2v_file)
    
    # Create Model with specified optimizer and loss function
    ##############################################################
    model = fastText(config, len(dataset.vocab), dataset.word_embeddings)
    if torch.cuda.is_available():
        model.cuda()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=config.lr)
    NLLLoss = nn.NLLLoss()
    model.add_optimizer(optimizer)
    model.add_loss_op(NLLLoss)
    ##############################################################
    
    train_losses = []
    val_accuracies = []
    best_val_f1 = 0.0
    best_epoch = 0
    best_train_f1= 0.0
    
    for i in range(config.max_epochs):
        print ("Epoch: {}".format(i))
        train_loss,val_accuracy = model.run_epoch(dataset.train_iterator, dataset.val_iterator, i)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)
        #Evaluate after each epoch
        train_f1 = evaluate_model(model, dataset.train_iterator)
        val_f1 = evaluate_model(model, dataset.val_iterator)
        print('Afrter epoch {}: Training F1: {:.5f} Validation F1: {:.4f}'.format(i+1,train_f1,val_f1))
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = i + 1
            best_train_f1 = train_f1
        W.write('Epoch:' + str(i+1)  + ' Train F1-score:' + str(train_f1) + ' Val F1-score:' + str(val_f1) + '\n')

    final_train_f1 = evaluate_model(model, dataset.train_iterator)
    final_val_f1 = evaluate_model(model, dataset.val_iterator)
    #test_acc = evaluate_model(model, dataset.test_iterator)

    print ('Final Training Accuracy: {:.4f}'.format(final_train_f1))
    print ('Final Validation Accuracy: {:.4f}'.format(final_val_f1))
    #print ('Final Test Accuracy: {:.4f}'.format(test_acc))
    print('Best validation score:{} at epoch:{}'.format(best_val_f1,best_epoch))
    W.write('Final Training F1-score:' + str(final_train_f1) + '\n')
    W.write('Final Validation F1-score:' + str(final_val_f1) + '\n')
    W.write('Best val-F1:' + str(best_val_f1) + ' Epoc: ' + str(best_epoch) + ' Train-F1:' + str(best_train_f1))
    W.close()
