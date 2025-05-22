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



def generate_transformer_text_greedy(model, start_string, char2idx, idx2char, length=200, device='cpu'):
    model.eval()
    input_ids = torch.tensor([char2idx[c] for c in start_string]).unsqueeze(0)
    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(length):
            if generated.size(1)>512:
                generated = generated[:,-512:] # truncate large context
            
            logits = model(generated) # shape (1,T, vocab_size)
            next_logits = logits[:, -1, :] # adjust temperature
            probs = F.softmax(next_logits, dim = -1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat((generated, next_token),dim=1)

    # Convert final tensor to string
    output = ''.join([idx2char[token.item()] for token in generated[0]])

    return output

def generate_transformer_text_temperature_scaling(model, start_string,temperature, char2idx, idx2char, length=200, device='cpu'):
    model.eval()
    input_ids = torch.tensor([char2idx[c] for c in start_string]).unsqueeze(0)
    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(length):
            if generated.size(1)>512:
                generated = generated[:,-512:] # truncate large context
            
            logits = model(generated) # shape (1,T, vocab_size)
            next_logits = logits[:, -1, :] / temperature # adjust temperature
            probs = F.softmax(next_logits, dim = -1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat((generated, next_token),dim=1)

    # Convert final tensor to string
    output = ''.join([idx2char[token.item()] for token in generated[0]])

    return output

def generate_transformer_text_nucleus_sampling(model, start_string,p, char2idx, idx2char, length=200, device='cpu'):
    model.eval()
    input_ids = torch.tensor([char2idx[c] for c in start_string]).unsqueeze(0)
    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(length):
            if generated.size(1)>512:
                generated = generated[:,-512:] # truncate large context
            
            logits = model(generated) # shape (1,T, vocab_size)
            next_logits = logits[:, -1, :]  # adjust temperature
            probs = torch.softmax(next_logits, dim = 1).squeeze()
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
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
            next_token = torch.multinomial(new_probs, num_samples=1).reshape((1,1))

            generated = torch.cat((generated, next_token),dim=1)

    # Convert final tensor to string
    output = ''.join([idx2char[token.item()] for token in generated[0]])

    return output