# %%
import torch
import numpy as np
import torch.nn as nn
import torch.utils
import torch.utils.data
import time

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

def create_dataset(text_as_int, seq_length, batch_size):
    total_num_seq = len(text_as_int) - seq_length
    inputs = []
    targets = []

    for i in range(0,total_num_seq):
        inputs.append(text_as_int[i:i+seq_length])
        targets.append(text_as_int[i+1:i+seq_length+1])
    
    inputs = torch.tensor(np.array(inputs))
    targets = torch.tensor(np.array(targets))
    
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=True, drop_last=True)

    return dataloader

# Testing code
'''
text = load_text()[:50]
vocab, char2idx, idx2char = create_vocab(text)
int_text = text_to_int(text, char2idx)
text_int = int_to_text(int_text, idx2char)
dataLoad = create_dataset(int_text, 20, 50)
for batch in dataLoad:
    inputs, targets = batch
    for inp, trg in zip(inputs,targets):
        print('--------------------\n')
        print(f'The input: {int_to_text(inp, idx2char)}, corresponds to the output {int_to_text(trg, idx2char)}\n')
        print('--------------------\n')
    break
'''

# %%
# Let's create all the basis for the RNN architecture implemented with PyTorch
# The size of the feture in nn.RNN(hidden_size, feature) was selected as 'hidden size'
#       for simplicity

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, seq_length):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.seq_legth = seq_length

        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden):
        x = self.embed(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)


# %%
def get_dataloader(seq_length, batch_size, path_file = 'shakespeare.txt', amount_chars = None):
    if amount_chars:
        text = load_text(path_file)[:amount_chars]
    else:
        text = load_text(path_file)
    print(f'Text of len {len(text)} is being processed.\n')
    vocab, char2idx, idx2char = create_vocab(text)
    text_as_int = text_to_int(text, char2idx)
    dataloader = create_dataset(text_as_int, seq_length, batch_size)

    return dataloader, vocab, char2idx, idx2char, text_as_int

# %%
# Let's see how this architecture works

'''
seq_length = 100
batch_size = 64
hidden_size = 128
epochs = 5
learning_rate = 0.003
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''

def train_RNN(seq_length, batch_size, hidden_size, epochs, learning_rate, device, amount_chars = None):
    
    # Get data
    dataloader, vocab, char2idx, idx2char, text_as_int = get_dataloader(seq_length, batch_size, amount_chars= amount_chars)

    # Model
    model = CharRNN(len(vocab), hidden_size, seq_length).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f'Training the RNN vanilla network.')

    initial_run_time = time.time()

    model.train()
    # Training
    for epoch in range(epochs):
        
        start_time = time.time()

        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            hidden = model.init_hidden(batch_size).to(device)
            optimizer.zero_grad()
            output, hidden = model(x_batch, hidden)
            loss = criterion(output.view(-1, len(vocab)), y_batch.view(-1))
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, batch duration {time.time() - start_time:.2f} seconds.\n')
    
    print(f'Total training time {time.time()-initial_run_time:.2f}.\n')


    return model, char2idx, idx2char

# %%
# Let's create a function to evaluate the model

def generat_text(model, start_string, char2idx, idx2char, length = 200, device = 'cpu', is_lstm = False):

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

# %%
# Some examples of training with a text[:10,000] and text[:100,000]

'''Generated text:

ROMEO: the would all as stand not was memost thenss the seake i' the love they beart, by their own resuar: the poor cive tare fo,
The couns to dognous,
That hunger wonche alarves Marcius.

First Citizen:
Ye'''

'''Generated text:

ROMEO: the would all as stand not was memost thenss the seake i' the love they beart, by their own resuar: the poor cive tare fo,
The couns to dognous,
That hunger wonche alarves Marcius.

First Citizen:
Ye'''

# %%
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=1):
        super(CharLSTM,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden
    
    def init_hidden(self, batch_size, device):
        # (num_layers, batch_size, hidden_size)
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h_0, c_0)

# %%
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

def train_lstm(seq_length, batch_size, hidden_size, epochs, learning_rate, device, layers, amount_chars = None):

    dataloader, vocab, char2idx, idx2char, text_as_int = get_dataloader(seq_length, batch_size, amount_chars=amount_chars)
    # Initialize model
    model_lstm = CharLSTM(len(vocab), hidden_size, num_layers= layers).to(device)
    optimizer = torch.optim.Adam(model_lstm.parameters(), lr = learning_rate)
    criterion = nn.CrossEntropyLoss()

    print(f'Training network LSTM of {layers} number of layers')

    initial_run_time = time.time()
    model_lstm.train()
    # Train
    for epoch in range(epochs):
        start_time = time.time()
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            hidden = model_lstm.init_hidden(batch_size, device)
            optimizer.zero_grad()
            output, hidden = model_lstm(x_batch, hidden)
            loss = criterion(output.view(-1, len(vocab)), y_batch.view(-1))
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, batch duration {time.time() - start_time:.2f} s')
    
    print(f'Total training time {time.time()-initial_run_time:.2f}.\n')
    
    return model_lstm, char2idx, idx2char 



'''
# %%
seq_length = 10
batch_size = 20
hidden_size = 128
epochs = 5
learning_rate = 0.003
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
layers_LSTM = 1

# Train the LSTM model
#model_lstm, char2idx, idx2char = train_lstm(seq_length, batch_size, hidden_size, epochs,learning_rate, device,layers=layers_LSTM, amount_chars=None)
#text = generat_text(model_lstm, start_string='ROMEO: ', char2idx= char2idx, idx2char=idx2char, device=device, is_lstm=True)
#print("\nGenerated text:\n")
#print(text)

# %%
# Train the vanilla RNN model
model_rnn, char2idx, idx2char = train_RNN(seq_length, batch_size, hidden_size, epochs, learning_rate, device, amount_chars=None)
text = generat_text(model_rnn, start_string='ROMEO: ', char2idx= char2idx, idx2char=idx2char, device=device)
print("\nGenerated text:\n")
print(text)

# %%
'''



