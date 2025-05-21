import torch
import numpy as np
import torch.nn as nn
import torch.utils
import torch.utils.data
import time
import torch.nn.functional as F
from RNN import CharRNN
from train_rnn_lstm import *
from generate import *
from metrics import *
nb_runs = 10
seq_length = 10
batch_size = 20
hidden_size = 128
epochs = 1
learning_rate = 0.003
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset,valid_dataset,test_dataset,vocab,char2idx,idx2char = create_dataset(seq_length,batch_size)
test_dataloader = get_dataloader(test_dataset,batch_size)
valid_dataloader = get_dataloader(valid_dataset,batch_size)


# Train the vanilla RNN model
"""
model_rnn_vanilla, char2idx, idx2char,losses = train_RNN(train_dataset,seq_length, batch_size, hidden_size, epochs, learning_rate, device,vocab,char2idx,idx2char)
text = generat_text_greedy(model_rnn_vanilla, start_string='ROMEO: ', char2idx= char2idx, idx2char=idx2char, device=device)
print("\nGenerated text with vanilla RNN:\n")
print(text)"""
"""
save_vanilla_rnn_path = "./models/vanilla_rnn.pth"
#torch.save(model_rnn_vanilla.state_dict(),save_vanilla_rnn_path)
model_rnn_vanilla = CharRNN(len(vocab),hidden_size,seq_length)
model_rnn_vanilla.load_state_dict(torch.load(save_vanilla_rnn_path))"""
# generate text geedily
"""
percentage_correctly_spelled_words_runs =[]
for iter in range(nb_runs):
    text = generat_text_greedy(model_rnn_vanilla, start_string='ROMEO: ',char2idx= char2idx, idx2char=idx2char, device=device)
    percentage_correctly_spelled_words_runs.append(percentage_correctly_spelled_words(text))
print( f" percentage of correctly spelt words vanilla rnn greedy sampling  : {sum(percentage_correctly_spelled_words_runs)/5}")
## perplexity on test set
print(f"perplexity vanilla rnn  greedy sampling : {peplexity_greedy(model_rnn_vanilla,test_dataloader,vocab,batch_size,device,is_lstm=False)}")
## 2 grams
## 3 grams
# Generate with temperature sampling
percentage_correctly_spelled_words_runs =[]
for iter in range(nb_runs):
    text = generat_text_with_temperature_sampling(model_rnn_vanilla, start_string='ROMEO: ',T = 0.9,char2idx= char2idx, idx2char=idx2char, device=device)
    percentage_correctly_spelled_words_runs.append(percentage_correctly_spelled_words(text))
print( f" percentage of correctly spelt words vanilla rnn greedy temperature sampling 0.9 : {sum(percentage_correctly_spelled_words_runs)/5}")

print(f"perplexity vanilla rnn  temperature  sampling T = 0.9 : {peplexity_temperature_scaling(model_rnn_vanilla,test_dataloader,vocab,batch_size,device,T=0.9,is_lstm=False)}")
"""
# Generate with nucleus sampling
"""
text = generat_text_nucleus_sampling(model_rnn_vanilla, start_string='ROMEO: ',p = 0.7,char2idx= char2idx, idx2char=idx2char, device=device)
print( f" percentage of correctly spelt words vanilla rnn nucleus sampling 0.7 : {percentage_correctly_spelled_words(text)}")
print(f"perplexity vanilla rnn  nucleus  sampling p = 0.7 : {peplexity_nucleus_sampling(model_rnn_vanilla,test_dataloader,vocab,batch_size,device,p=0.7,is_lstm=False)}")
"""
# Train one layer LSTM
"""
text = generat_text_with_temperature_sampling(model_rnn_vanilla, start_string='ROMEO: ',T = 0.1,char2idx= char2idx, idx2char=idx2char, device=device)
print( f" percentage of correctly spelt words vanilla rnn greedy temperature sampling 0.1 : {percentage_correctly_spelled_words(text)}")
# Generate with nucleus sampling"""
"""
layers_LSTM = 1
model_lstm, char2idx, idx2char,losses_lstm_one_layer = train_lstm(train_dataset,seq_length, batch_size, hidden_size, epochs, learning_rate, device, layers_LSTM,vocab,char2idx,idx2char)
# Generate greedily
text = generat_text_greedy(model_lstm, start_string='ROMEO: ', char2idx= char2idx, idx2char=idx2char, device=device, is_lstm=True)
print("\nGenerated text:\n")
print(text)
# Generate with nucleus sampling"""
save_LSTM_one_layer_path = "./models/lstm_1.pth"
#torch.save(model_rnn_vanilla.state_dict(),save_vanilla_rnn_path)
model_lstm_one_layer = CharLSTM(len(vocab),hidden_size,1)
model_lstm_one_layer.load_state_dict(torch.load(save_LSTM_one_layer_path))
"""
percentage_correctly_spelled_words_runs =[]
for iter in range(nb_runs):
    text = generat_text_greedy(model_lstm_one_layer, start_string='ROMEO: ',char2idx= char2idx, idx2char=idx2char, device=device,is_lstm=True)
    percentage_correctly_spelled_words_runs.append(percentage_correctly_spelled_words(text))
print( f" percentage of correctly spelt words one layer greedy : {sum(percentage_correctly_spelled_words_runs)/nb_runs}")

print(f"perplexity one layer LSTM  greedy : {peplexity_greedy(model_lstm_one_layer,test_dataloader,vocab,batch_size,device,is_lstm=True)}")"""
"""
percentage_correctly_spelled_words_runs =[]
for iter in range(nb_runs):
    text = generat_text_with_temperature_sampling(model_lstm_one_layer, start_string='ROMEO: ',T = 0.9,char2idx= char2idx, idx2char=idx2char, device=device,is_lstm=True)
    percentage_correctly_spelled_words_runs.append(percentage_correctly_spelled_words(text))
print( f" percentage of correctly spelt words one layer LSTM  temperature sampling 0.9 : {sum(percentage_correctly_spelled_words_runs)/nb_runs}")

print(f"perplexity one layer LSTM  temperature  sampling T = 0.9 : {peplexity_temperature_scaling(model_lstm_one_layer,test_dataloader,vocab,batch_size,device,T=0.9,is_lstm=True)}")"""
percentage_correctly_spelled_words_runs =[]
for iter in range(nb_runs):
    text = generat_text_nucleus_sampling(model_lstm_one_layer, start_string='ROMEO: ',p = 0.7,char2idx= char2idx, idx2char=idx2char, device=device,is_lstm=True)
    percentage_correctly_spelled_words_runs.append(percentage_correctly_spelled_words(text))
print( f" percentage of correctly spelt words one layer greedy : {sum(percentage_correctly_spelled_words_runs)/nb_runs}")

print(f"perplexity one layer LSTM  greedy : {peplexity_nucleus_sampling(model_lstm_one_layer,test_dataloader,vocab,batch_size,device,p = 0.7,is_lstm=True)}")