import torch
import numpy as np
import torch.nn as nn
import torch.utils
from  torch.utils.data import random_split,DataLoader,TensorDataset
import time
import torch.nn.functional as F
from RNN import *
from LSTM import *
from generate import *


#### Load data and create datasets
def load_text(file_path = 'shakespeare.txt'):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def create_vocab(text):
    vocab = sorted(set(text))
    char2idx = {char_i:i for i, char_i in enumerate(vocab)}
    idx2char = np.array(vocab)
    return vocab, char2idx, idx2char

def text_to_int(text, char2idx):
    return np.array([char2idx[i] for i in text])

def int_to_text(index, idx2char):
    return ''.join(idx2char[index])

def create_dataset(text_as_int, seq_length,amount_chars=None):
    if amount_chars:
        text = load_text("./shakespeare.txt")[:amount_chars]
    else:
        text = load_text("./shakespeare.txt")
    print(f'Text of len {len(text)} is being processed.\n')
    vocab, char2idx, idx2char = create_vocab(text)
    text_as_int = text_to_int(text, char2idx)
    total_num_seq = len(text_as_int) - seq_length
    inputs = []
    targets = []

    for i in range(0,total_num_seq):
        inputs.append(text_as_int[i:i+seq_length])
        targets.append(text_as_int[i+1:i+seq_length+1])

    inputs = torch.tensor(np.array(inputs))
    targets = torch.tensor(np.array(targets))

    dataset = TensorDataset(inputs, targets)
    train_size = int(0.9 * len(dataset))   # 90% for training
    valid_size = int(0.05 * len(dataset))   # 5% for validation
    test_size = len(dataset) - train_size - valid_size  # Remaining 5% for testing
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])


    return train_dataset,valid_dataset,test_dataset,vocab,char2idx,idx2char

def get_dataloader(dataset, batch_size):
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return dataloader




def train_RNN(dataset,seq_length, batch_size, hidden_size, epochs, learning_rate, device, vocab,char2idx,idx2char):

    # Get data
    dataloader = get_dataloader(dataset,batch_size)

    # Model
    model = CharRNN(len(vocab), hidden_size, seq_length).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    step = 0

    print(f'Training the RNN vanilla network.')

    initial_run_time = time.time()

    # Training
    for epoch in range(epochs):

        start_time = time.time()

        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            hidden = model.init_hidden(batch_size).to(device)
            optimizer.zero_grad()
            output, hidden = model(x_batch, hidden)
            loss = criterion(output.view(-1, len(vocab)), y_batch.view(-1))
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            if (step%1000)==0:
                print(step)
                text = generat_text_greedy(model, start_string='ROMEO: ', char2idx= char2idx, idx2char=idx2char, device=device)
                print(f"Iter : {step}")
                print("generated text : ")
                print(text)
                save_path = "./models/vanilla_rnn.pth"
                torch.save(model.state_dict(),save_path)
            step+=1
            if step>=2000:
                break


        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, batch duration {time.time() - start_time:.2f} seconds.\n')

    print(f'Total training time {time.time()-initial_run_time:.2f}.\n')


    return model, char2idx, idx2char,losses



# Let's train the LSTM model

'''
seq_length = 10
batch_size = 20
hidden_size = 128
epochs = 5
learning_rate = 0.003
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
layers = 1
'''

def train_lstm(dataset,seq_length, batch_size, hidden_size, epochs, learning_rate, device, layers,vocab,char2idx,idx2char):

    dataloader = get_dataloader(dataset,batch_size)
    # Initialize model
    model_lstm = CharLSTM(len(vocab), hidden_size, num_layers= layers).to(device)
    optimizer = torch.optim.Adam(model_lstm.parameters(), lr = learning_rate)
    criterion = nn.CrossEntropyLoss()
    losses = []
    step = 0

    print(f'Training network LSTM of {layers} number of layers')

    initial_run_time = time.time()
    # Train
    for epoch in range(epochs):
        start_time = time.time()
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            hidden = model_lstm.init_hidden(batch_size, device)
            optimizer.zero_grad()
            output, hidden = model_lstm(x_batch, hidden)
            loss = criterion(output.view(-1, len(vocab)), y_batch.view(-1))
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            if (step%1000)==0:
                model_lstm.eval()
                print(step)
                text = generat_text_greedy(model_lstm, start_string='ROMEO: ', char2idx= char2idx, idx2char=idx2char, device=device,is_lstm=True)
                print(f"Iter : {step}")
                print("generated text : ")
                print(text)
                save_path = f"./models/lstm_{layers}.pth"
                torch.save(model_lstm.state_dict(),save_path)
            step+=1
            if (step>=2000):
                break

        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, batch duration {time.time() - start_time:.2f} s')

    print(f'Total training time {time.time()-initial_run_time:.2f}.\n')

    return model_lstm, char2idx, idx2char,losses


