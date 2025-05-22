import torch
import numpy as np
import torch.nn as nn
import torch.utils
import torch.utils.data
import time
import torch.nn.functional as F
from RNN import CharRNN

def generat_text_greedy(model, start_string, char2idx, idx2char, length = 200, device = 'cpu', is_lstm = False):

    model.eval()
    input_eval = torch.tensor([char2idx[i] for i in start_string]).unsqueeze(0).to(device)

    if not is_lstm:
        hidden = model.init_hidden(1).to(device)
    elif is_lstm:
        hidden = model.init_hidden(1, device)

    generated = list(start_string)

    with torch.no_grad():
        for i in range(length):

            output, hidden = model(input_eval, hidden)
            logits = output[:,-1, :] # In this line we can add temperature
            probs = torch.softmax(logits, dim = 1).squeeze()


            next_idx = torch.multinomial(probs,1).item()
            next_char = idx2char[next_idx]

            generated.append(next_char)

            input_eval = torch.tensor([[next_idx]]).to(device)

    return ''.join(generated)

# Let's create a function to evaluate the model

def generat_text_with_temperature_sampling(model, start_string,T, char2idx, idx2char, length = 200, device = 'cpu', is_lstm = False):

    model.eval()
    input_eval = torch.tensor([char2idx[i] for i in start_string]).unsqueeze(0).to(device)

    if not is_lstm:
        hidden = model.init_hidden(1).to(device)
    elif is_lstm:
        hidden = model.init_hidden(1, device)

    generated = list(start_string)

    with torch.no_grad():
        for i in range(length):

            output, hidden = model(input_eval, hidden)
            logits = output[:,-1, :] # In this line we can add temperature
            probs = torch.softmax(logits/T, dim = 1).squeeze()


            next_idx = torch.multinomial(probs,1).item()
            next_char = idx2char[next_idx]

            generated.append(next_char)

            input_eval = torch.tensor([[next_idx]]).to(device)

    return ''.join(generated)


# Let's create a function to evaluate the model

def generat_text_nucleus_sampling(model, start_string,p, char2idx, idx2char, length = 200, device = 'cpu', is_lstm = False):

    model.eval()
    input_eval = torch.tensor([char2idx[i] for i in start_string]).unsqueeze(0).to(device)
    log_probs = []

    if not is_lstm:
        hidden = model.init_hidden(1).to(device)
    elif is_lstm:
        hidden = model.init_hidden(1, device)

    generated = list(start_string)

    with torch.no_grad():
        for i in range(length):

            output, hidden = model(input_eval, hidden)
            logits = output[:,-1, :] # In this line we can add temperature
            probs = torch.softmax(logits, dim = 1).squeeze()
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = F.softmax(sorted_logits,dim=-1)
            new_probs = torch.zeros_like(probs)
            cum = 0
            index = 0
            for v in range(sorted_logits.size(1)):
                cum+=sorted_probs[0,v:v+1]
                index=v
                if (cum>=p):
                    break
            new_probs[sorted_indices[0,:index+1]]=probs[sorted_indices[0,:index+1]]/cum


            next_idx = torch.multinomial(new_probs,1).item()
            next_char = idx2char[next_idx]

            generated.append(next_char)

            input_eval = torch.tensor([[next_idx]]).to(device)

    return ''.join(generated)



