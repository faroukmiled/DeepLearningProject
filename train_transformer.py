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
from nltk.translate.bleu_score import sentence_bleu
import random
import matplotlib.pyplot as plt
import functions_main as fm
from transformer import *
def compute_loss_transformer(model,valid_dataloader,batch_size,vocab,device,is_lstm=False):
    losses = []
    criterion = nn.CrossEntropyLoss()
    model.eval()
    for x_batch, y_batch in valid_dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)
            loss = criterion(output.view(-1, len(vocab)), y_batch.view(-1))
            losses.append(loss.item())
    return sum(losses)/len(losses)
# Setup
seq_length = 30
batch_size = 200
hidden_size = 64
epochs = 1
learning_rate = 0.003
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
layers_LSTM = 1
amount_chars = None

# Training block
train_dataset,valid_dataset,test_dataset,vocab,char2idx,idx2char = create_dataset(seq_length,batch_size)
test_dataloader = get_dataloader(test_dataset,batch_size)
valid_dataloader = get_dataloader(valid_dataset,batch_size)
train_dataloader = get_dataloader(valid_dataset,batch_size)
text = load_text()
save_path = f"./models/transfomer.pth"
# Generate with nucleus sampling"""
ref_words  =  " ".join(get_words_from_text(text))
model = CharTransformer(vocab_size=len(vocab)).to(device)
model.load_state_dict(torch.load(save_path))
"""
optimizer = torch.optim.Adam(model.parameters(), learning_rate)
criterion = nn.CrossEntropyLoss()

start_time = time.time()
train_losses = []
valid_losses = []
model.train()
step = 0
for epoch in range(epochs):
    start_epoch = time.time()
    for x_batch, y_batch in train_dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        logits = model(x_batch)
        loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
        if step%50==0:
            train_losses.append(loss.item())
            valid_losses.append(compute_loss_transformer(model,valid_dataloader,batch_size,vocab,device))
        loss.backward()
        optimizer.step()
        step+=1
        print(step)
        if step%100==0:
             save_path = f"./models/transfomer.pth"
             torch.save(model.state_dict(),save_path)
        if step>=3000:
             break
    end_epoch = time.time()
    epoch_time = end_epoch - start_epoch
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}, Total time past {epoch_time:.2f}')
print(f'Model trained in {time.time() - start_time:.2f}')

plt.plot(np.arange(len(train_losses))*100,train_losses,label="training loss")
plt.plot(np.arange(len(valid_losses))*100,valid_losses,label="validation, loss")
plt.xlabel("update step")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss transformer.png")"""
gen_text = generate_transformer_text_nucleus_sampling(model,start_string="ROMEO :",p=0.7,char2idx = char2idx,idx2char =idx2char,length=10000,device=device)
print(gen_text)
gen_words =  " ".join(get_words_from_text(gen_text))
two_grams_score = n_grames_metric(gen_words,ref_words,n=2)
three_grams_score = n_grames_metric(gen_words,ref_words,n=3)
print(f" two grams score {two_grams_score}")
print(f" three grams score {three_grams_score}")
print(f" percentage words : {percentage_correctly_spelled_words(get_words_from_text(gen_words))}")
#print(f" perplexity greedy :{peplexity_nucleus_sampling(model,test_dataloader,vocab,batch_size=batch_size,device=device,p=0.7,is_lstm=False)} ")
print(" _---- ------")
"""
text_temp = generate_transformer_text_temperature_scaling(model,start_string="ROMEO :",temperature=0.9,char2idx = char2idx,idx2char =idx2char,device=device)
print(text_temp)
#text_greedy = generate_transformer_text_greedy(model,start_string="ROMEO :",char2idx = char2idx,idx2char =idx2char,device=device)
print("------")
text_nuc= generate_transformer_text_nucleus_sampling(model,start_string="ROMEO :",p=0.7,char2idx = char2idx,idx2char =idx2char,device=device)
print(text_nuc)"""