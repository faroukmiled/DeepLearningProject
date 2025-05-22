import torch
import numpy as np
import torch.nn as nn
import torch.utils
import torch.utils.data
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
from RNN import CharRNN
from train_rnn_lstm import *
from generate import *
from metrics import *
from nltk.translate.bleu_score import sentence_bleu
import random
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
text = load_text()

# Train the vanilla RNN model
"""
model_rnn_vanilla, char2idx, idx2char,losses = train_RNN(train_dataset,seq_length, batch_size, hidden_size, epochs, learning_rate, device,vocab,char2idx,idx2char)
text = generat_text_greedy(model_rnn_vanilla, start_string='ROMEO: ', char2idx= char2idx, idx2char=idx2char, device=device)
print("\nGenerated text with vanilla RNN:\n")
print(text)"""
save_vanilla_rnn_path = "./models/vanilla_rnn.pth"
#torch.save(model_rnn_vanilla.state_dict(),save_vanilla_rnn_path)
model_rnn_vanilla = CharRNN(len(vocab),hidden_size,seq_length)
model_rnn_vanilla.load_state_dict(torch.load(save_vanilla_rnn_path))
'''save_LSTM_one_layer_path = "./models/lstm_1.pth"
#torch.save(model_rnn_vanilla.state_dict(),save_vanilla_rnn_path)
model_lstm_one_layer = CharLSTM(len(vocab),hidden_size,1)
model_lstm_one_layer.load_state_dict(torch.load(save_LSTM_one_layer_path))'''
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
#start_string = text[:110]
#gen_text = generat_text_greedy(model_rnn_vanilla, start_string=start_string,char2idx= char2idx, idx2char=idx2char, device=device)
#ref_text = text[110:310]
#print(BLEU(gen_text,ref_text))
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
ref_words  =  " ".join(get_words_from_text(text))
learning_rates = np.linspace(0.0001, 0.01, 100)  # 100 values from 0.0001 to 0.01
batch_sizes = [16, 32, 64]  # Example batch sizes

# Random search parameters
num_trials = 7  # Number of random trials

# Store results
results = []

for _ in range(num_trials):
    lr = random.choice(learning_rates)  # Randomly select learning rate
    batch_size = random.choice(batch_sizes)  # Randomly select batch size
    train_dataset,valid_dataset,test_dataset,vocab,char2idx,idx2char = create_dataset(seq_length,batch_size)
    test_dataloader = get_dataloader(test_dataset,batch_size)
    valid_dataloader = get_dataloader(valid_dataset,batch_size)
    
    # Simulate training (replace this with actual model training)
    print(f"Training with learning rate: {lr:.4f} and batch size: {batch_size}")
    
    # Here you would call your training function and get the performance
    # For demonstration, we simulate a random accuracy
    model_one_layer_lstm, char2idx, idx2char,train_losses,valid_losses = train_lstm(train_dataset,valid_dataset,seq_length, batch_size, hidden_size, epochs, learning_rate, device,1,vocab,char2idx,idx2char)
    ppl = peplexity_nucleus_sampling(model_one_layer_lstm,valid_dataloader,vocab,batch_size,device,p=0.7,is_lstm=True)  # Simulated accuracy between 0.5 and 1.0
    print(f"Learning Rate: {lr:.4f}, Batch Size: {batch_size}, Accuracy: {ppl:.4f}")
    results.append((lr,batch_size,ppl))

# Print results
for lr, bs, ppl in results:
    print(f"Learning Rate: {lr:.4f}, Batch Size: {bs}, Accuracy: {ppl:.4f}")
for index,hidden_size in enumerate([128]):
    #model_lstm_two_layer, char2idx, idx2char,train_losses,valid_losses = train_lstm(train_dataset,valid_dataset,seq_length, batch_size, hidden_size, epochs, learning_rate, device,2,vocab,char2idx,idx2char)
    """gen_text = generat_text_greedy(model_rnn_vanilla, start_string='ROMEO: ', char2idx= char2idx, idx2char=idx2char,length=10000, device=device)
    print(f"text_generated_greedy : \n {generat_text_greedy(model_rnn_vanilla, start_string='ROMEO: ', char2idx= char2idx, idx2char=idx2char,length=200, device=device)}")
    gen_words =  " ".join(get_words_from_text(gen_text))
    two_grams_score = n_grames_metric(gen_words,ref_words,n=2)
    three_grams_score = n_grames_metric(gen_words,ref_words,n=3)
    print(f"rnn {hidden_size} two grams score {two_grams_score}")
    print(f"rnn {hidden_size} three grams score {three_grams_score}")
    print(f"rnn {hidden_size} percentage words : {percentage_correctly_spelled_words(get_words_from_text(gen_words))}")
    print(f"rnn {hidden_size} perplexity greedy :{peplexity_greedy(model_rnn_vanilla,test_dataloader,vocab,batch_size=batch_size,device=device,is_lstm=False)} ")
    print("-------------------")
    gen_text = generat_text_with_temperature_sampling(model_rnn_vanilla, start_string='ROMEO: ',T=0.9, char2idx= char2idx, idx2char=idx2char, length=10000,device=device)
    print(f"text_generated_temperature : \n {generat_text_with_temperature_sampling(model_rnn_vanilla, start_string='ROMEO: ',T=0.9, char2idx= char2idx, idx2char=idx2char,length=200, device=device)}")
    gen_words =  " ".join(get_words_from_text(gen_text))
    two_grams_score = n_grames_metric(gen_words,ref_words,n=2)
    three_grams_score = n_grames_metric(gen_words,ref_words,n=3)
    print(f"rnn {hidden_size} two grams score {two_grams_score}")
    print(f"rnn {hidden_size} three grams score {three_grams_score}")
    print(f"rnn {hidden_size} percentage words : {percentage_correctly_spelled_words(get_words_from_text(gen_words))}")
    print(f"rnn {hidden_size} perplexity greedy :{peplexity_temperature_scaling(model_rnn_vanilla,test_dataloader,vocab,batch_size=batch_size,device=device,T=0.9,is_lstm=False)} ")"""
    print('--------------------')
    #gen_text = generat_text_nucleus_sampling(model_lstm_two_layer, start_string='ROMEO: ',p=0.7, char2idx= char2idx, idx2char=idx2char, length=10000,device=device,is_lstm=True)
    print(f"text_nucleus_sampling : \n {generat_text_nucleus_sampling(model_lstm_two_layer, start_string='ROMEO: ',p=0.7, char2idx= char2idx, idx2char=idx2char,length=200, device=device,is_lstm=True)}")
    #gen_words =  " ".join(get_words_from_text(gen_text))
    #two_grams_score = n_grames_metric(gen_words,ref_words,n=2)
    #three_grams_score = n_grames_metric(gen_words,ref_words,n=3)
    #print(f"rnn {hidden_size} two grams score {two_grams_score}")
    #print(f"rnn {hidden_size} three grams score {three_grams_score}")
    #print(f"rnn {hidden_size} percentage words : {percentage_correctly_spelled_words(get_words_from_text(gen_words))}")
    #print(f"rnn {hidden_size} perplexity greedy :{peplexity_nucleus_sampling(model_lstm_two_layer,test_dataloader,vocab,batch_size=batch_size,device=device,p=0.7,is_lstm=True)} ")

    """plt.figure(index)
    plt.plot(np.arange(0,3000,100),valid_losses,label="training loss")
    plt.plot(np.arange(0,3000,100),train_losses,label ="validation loss")
    plt.xlabel("update step")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("rnn  512 loss.png")"""
"""
save_LSTM_one_layer_path = "./models/lstm_1.pth"
#torch.save(model_rnn_vanilla.state_dict(),save_vanilla_rnn_path)
model_lstm_one_layer = CharLSTM(len(vocab),hidden_size,1)
model_lstm_one_layer.load_state_dict(torch.load(save_LSTM_one_layer_path))
gen_text = generat_text_greedy(model_rnn_vanilla, start_string=start_string,char2idx= char2idx, idx2char=idx2char,length=10000, device=device,is_lstm=False)
gen_words =  " ".join(get_words_from_text(gen_text))
ref_words  =  " ".join(get_words_from_text(text))
two_grams_score = n_grames_metric(gen_words,ref_words,n=2)
three_grams_score = n_grames_metric(gen_words,ref_words,n=3)
print(two_grams_score)
print(three_grams_score)
print(percentage_correctly_spelled_words(get_words_from_text(gen_words)))
save_LSTM_two_layer_path = "./models/lstm_2.pth"
#torch.save(model_rnn_vanilla.state_dict(),save_vanilla_rnn_path)
model_lstm_two_layer = CharLSTM(len(vocab),hidden_size,2)
model_lstm_two_layer.load_state_dict(torch.load(save_LSTM_two_layer_path))
gen_text = generat_text_with_temperature_sampling(model_rnn_vanilla, start_string=start_string,T=0.9,char2idx= char2idx, idx2char=idx2char,length=10000, device=device,is_lstm=False)
gen_words =  " ".join(get_words_from_text(gen_text))
ref_words  =  " ".join(get_words_from_text(text))
two_grams_score = n_grames_metric(gen_words,ref_words,n=2)
three_grams_score = n_grames_metric(gen_words,ref_words,n=3)
print(two_grams_score)
print(three_grams_score)
print(percentage_correctly_spelled_words(get_words_from_text(gen_words)))"""
"""
model_lstm_two_layer = CharLSTM(len(vocab),hidden_size,2)
model_lstm_two_layer.load_state_dict(torch.load(save_LSTM_two_layer_path))
gen_text = generat_text_greedy(model_rnn_vanilla, start_string=start_string,char2idx= char2idx, idx2char=idx2char,length=10000, device=device,is_lstm=False)
gen_words =  " ".join(get_words_from_text(gen_text))
ref_words  =  " ".join(get_words_from_text(text))
two_grams_score = n_grames_metric(gen_words,ref_words,n=2)
three_grams_score = n_grames_metric(gen_words,ref_words,n=3)
print(two_grams_score)
print(three_grams_score)"""
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
"""
percentage_correctly_spelled_words_runs =[]
for iter in range(nb_runs):
    text = generat_text_nucleus_sampling(model_lstm_one_layer, start_string='ROMEO: ',p = 0.7,char2idx= char2idx, idx2char=idx2char, device=device,is_lstm=True)
    percentage_correctly_spelled_words_runs.append(percentage_correctly_spelled_words(text))
print( f" percentage of correctly spelt words one layer greedy : {sum(percentage_correctly_spelled_words_runs)/nb_runs}")

print(f"perplexity one layer LSTM  greedy : {peplexity_nucleus_sampling(model_lstm_one_layer,test_dataloader,vocab,batch_size,device,p = 0.7,is_lstm=True)}")"""