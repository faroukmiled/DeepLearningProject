import torch
import numpy as np
import torch.nn as nn
import torch.utils
from RNN import *
from LSTM import *
from utils import *
import nltk
import torch.nn.functional as F
from nltk.corpus import brown
from nltk import ngrams
from nltk.tokenize import word_tokenize
#nltk.download()

def peplexity_greedy(model,dataloader,vocab,batch_size,device,is_lstm=False):
    
    if not is_lstm:
        hidden = model.init_hidden(batch_size).to(device)
    elif is_lstm:
        hidden = model.init_hidden(batch_size, device)
    criterion = nn.CrossEntropyLoss()
    model.eval()
    losses = torch.zeros((len(dataloader),))
    step = 0
    for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output, hidden = model(x_batch, hidden)
            loss = criterion(output.view(-1, len(vocab)), y_batch.view(-1))
            losses[step]=loss
            step+=1
    return torch.exp(torch.mean(losses)).detach().numpy()

def peplexity_temperature_scaling(model,dataloader,vocab,batch_size,device,T,is_lstm=False):
    
    if not is_lstm:
        hidden = model.init_hidden(batch_size).to(device)
    elif is_lstm:
        hidden = model.init_hidden(batch_size, device)
    criterion = nn.CrossEntropyLoss()
    model.eval()
    losses = torch.zeros((len(dataloader),))
    step = 0
    for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output, hidden = model(x_batch, hidden)
            output = output/T
            loss = criterion(output.view(-1, len(vocab)), y_batch.view(-1))
            losses[step]=loss
            step+=1
    return torch.exp(torch.mean(losses)).detach().numpy()


def peplexity_nucleus_sampling(model,dataloader,vocab,batch_size,device,p,is_lstm=False):
    
    if not is_lstm:
        hidden = model.init_hidden(batch_size).to(device)
    elif is_lstm:
        hidden = model.init_hidden(batch_size, device)
    criterion = nn.NLLLoss()
    model.eval()
    losses = torch.zeros((len(dataloader),))
    step = 0
    for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output, hidden = model(x_batch, hidden)
            output = output
            logits = output[:,-1, :] # In this line we can add temperature
            samp_probs = torch.softmax(logits, dim = 1).squeeze()
            sorted_probs, sorted_indices = torch.sort(samp_probs, descending=True)
            #sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            sorted_samp_probs = sorted_probs.clone()
            sorted_samp_probs[sorted_indices_to_remove] = 0
            sorted_next_indices = sorted_samp_probs.multinomial(1).view(-1, 1)
            next_tokens = sorted_indices.gather(1, sorted_next_indices)
            next_logprobs = sorted_samp_probs.gather(1, sorted_next_indices).log()
            losses[step]=-torch.mean(next_logprobs)
            step+=1
    return torch.exp(torch.mean(losses)).detach().numpy()




def percentage_correctly_spelled_words(words_from_text):
    word_list = brown.words()
    word_set = set(word_list)
    nb = 0
    for word in words_from_text:
        if word in word_set:
          nb+=1
    return 100*nb/len(words_from_text)

def n_grames_metric(gen_text,ref_text,n=1):
     gen_tokens = word_tokenize(gen_text)
     ref_tokens = word_tokenize(ref_text)
     gen_ngrams = set(ngrams(gen_tokens, n))
     ref_ngrams = set(ngrams(ref_tokens, n))
     overlap = gen_ngrams.intersection(ref_ngrams)
     return len(overlap)/len(gen_ngrams)

def BLEU(gen_text,ref_text):
     res = min(1,len(gen_text)/len(ref_text)) 
     for n in range(1,2):
        precision = n_grames_metric(" ".join(gen_text)," ".join(ref_text),n)
        res*=precision
     return res

