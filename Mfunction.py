import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import random
from collections import defaultdict
from tqdm import tqdm

# Import our metrics and generation functions
from text_quality_metrics import TextQualityMetrics
from text_generation_functions import (
    generate_text_with_temperature, 
    generate_text_with_nucleus,
    compare_sampling_methods
)

# Set random seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Function to load and preprocess text data
def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

# Function to create character mappings
def create_char_mappings(text):
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return chars, char_to_idx, idx_to_char

# Function to prepare data for training
def prepare_data(text, char_to_idx, seq_length=100, batch_size=64):
    # Create input and target sequences
    data_size = len(text)
    num_batches = data_size // (seq_length * batch_size)
    
    # Truncate text to fit batches exactly
    text = text[:num_batches * seq_length * batch_size]
    
    # Vectorize the text
    char_indices = [char_to_idx[char] for char in text]
    x = np.array(char_indices[:-1])
    y = np.array(char_indices[1:])
    
    # Reshape into batches
    x_batches = np.split(x[:num_batches * batch_size * seq_length], batch_size)
    y_batches = np.split(y[:num_batches * batch_size * seq_length], batch_size)
    
    return x_batches, y_batches

# Create training, validation, and test splits
def create_data_splits(text, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"
    
    # Calculate split indices
    train_end = int(len(text) * train_ratio)
    val_end = train_end + int(len(text) * val_ratio)
    
    # Split the text
    train_text = text[:train_end]
    val_text = text[train_end:val_end]
    test_text = text[val_end:]
    
    return train_text, val_text, test_text

# Create data loader for training
def create_data_loader(x_batches, y_batches, batch_size=64, seq_length=100):
    """
    Create a PyTorch data loader from the batched data.
    """
    # Convert data to PyTorch tensors
    x_tensors = [torch.tensor(x.reshape(-1, seq_length), dtype=torch.long) for x in x_batches]
    y_tensors = [torch.tensor(y.reshape(-1, seq_length), dtype=torch.long) for y in y_batches]
    
    # Combine tensors
    x_tensor = torch.cat(x_tensors, dim=0)
    y_tensor = torch.cat(y_tensors, dim=0)
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    
    # Create data loader
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return loader

# Define the vanilla RNN model
class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
        
        # Process input through embedding layer
        x = self.embedding(x)
        
        # Pass through RNN
        output, hidden = self.rnn(x, hidden)
        
        # Pass through fully connected layer
        output = self.fc(output)
        
        return output, hidden

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, 
                           dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        # Initialize hidden state and cell state if not provided
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
            hidden = (h0, c0)
        
        # Process input through embedding layer
        x = self.embedding(x)
        
        # Pass through LSTM
        output, hidden = self.lstm(x, hidden)
        
        # Pass through fully connected layer
        output = self.fc(output)
        
        return output, hidden

# Define the GRU model (for C/D grade extension)
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, 
                         dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        # Process input through embedding layer
        x = self.embedding(x)
        
        # Pass through GRU
        output, hidden = self.gru(x, hidden)
        
        # Pass through fully connected layer
        output = self.fc(output)
        
        return output, hidden

# Training function
def train_model(model, data_loader, optimizer, criterion, device, clip=5.0):
    model.train()
    total_loss = 0
    
    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output, _ = model(x_batch)
        
        # Reshape output and target for loss calculation
        output = output.reshape(-1, output.shape[-1])
        y_batch = y_batch.reshape(-1)
        
        # Calculate loss
        loss = criterion(output, y_batch)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # Update weights
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(data_loader)

# Evaluation function
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass
            output, _ = model(x_batch)
            
            # Reshape output and target for loss calculation
            output = output.reshape(-1, output.shape[-1])
            y_batch = y_batch.reshape(-1)
            
            # Calculate loss
            loss = criterion(output, y_batch)
            total_loss += loss.item()
    
    return total_loss / len(data_loader)

# Calculate perplexity from loss
def calculate_perplexity(loss):
    return np.exp(loss)

# Plot training and validation loss
def plot_learning_curves(train_losses, val_losses, model_name, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Learning Curves for {model_name}')
    plt.legend()
    
    if save_path:
        plt.savefig(f"{save_path}/{model_name}_learning_curves.png")
    else:
        plt.show()

# Grid search for hyperparameters
def grid_search(model_class, train_loader, val_loader, vocabulary_size, 
                hidden_sizes=[128, 256, 512], 
                learning_rates=[0.001, 0.01], 
                batch_sizes=[32, 64, 128],
                num_epochs=10, device='cpu'):
    """
    Perform grid search to find optimal hyperparameters.
    """
    results = []
    
    for hidden_size in hidden_sizes:
        for lr in learning_rates:
            for batch_size in batch_sizes:
                print(f"Training with hidden_size={hidden_size}, lr={lr}, batch_size={batch_size}")
                
                # Create model
                model = model_class(vocabulary_size, hidden_size, vocabulary_size).to(device)
                
                # Define optimizer and criterion
                optimizer = optim.Adam(model.parameters(), lr=lr)
                criterion = nn.CrossEntropyLoss()
                
                # Train for a few epochs
                best_val_loss = float('inf')
                
                for epoch in range(num_epochs):
                    # Train
                    train_loss = train_model(model, train_loader, optimizer, criterion, device)
                    
                    # Evaluate
                    val_loss = evaluate_model(model, val_loader, criterion, device)
                    
                    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                    
                    # Update best validation loss
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                
                # Record results
                results.append({
                    'hidden_size': hidden_size,
                    'learning_rate': lr,
                    'batch_size': batch_size,
                    'best_val_loss': best_val_loss,
                    'perplexity': calculate_perplexity(best_val_loss)
                })
    
    # Sort results by validation loss
    results.sort(key=lambda x: x['best_val_loss'])
    
    return results

# Function to evaluate text quality across models
def evaluate_text_quality(models_dict, seed_text, char_to_idx, idx_to_char, 
                         reference_text, temperatures=[0.7, 1.0, 1.3], 
                         nucleus_p_values=[0.5, 0.8, 0.9], max_length=500, device='cpu'):
    """
    Evaluate text quality generated by different models with different sampling methods.
    """
    # Initialize metrics calculator
    metrics = TextQualityMetrics(reference_text=reference_text)
    
    # Results dictionary
    results = defaultdict(dict)
    
    # Evaluate each model
    for model_name, model in models_dict.items():
        print(f"Evaluating {model_name}...")
        
        # Temperature sampling evaluation
        temp_results = {}
        for temp in temperatures:
            generated_text = generate_text_with_temperature(
                model, seed_text, char_to_idx, idx_to_char, 
                max_length=max_length, temperature=temp, device=device
            )
            
            temp_results[temp] = {
                'text': generated_text,
                'lexical_diversity': metrics.lexical_diversity(generated_text),
                'bigram_overlap': metrics.n_gram_overlap(generated_text, n=2),
                'trigram_overlap': metrics.n_gram_overlap(generated_text, n=3),
                'spelling_accuracy': metrics.spell_check_percentage(generated_text),
                'repetition_ratio': metrics.repetition_ratio(generated_text)
            }
        
        results[model_name]['temperature'] = temp_results
        
        # Nucleus sampling evaluation
        nucleus_results = {}
        for p in nucleus_p_values:
            generated_text = generate_text_with_nucleus(
                model, seed_text, char_to_idx, idx_to_char, 
                max_length=max_length, p=p, device=device
            )
            
            nucleus_results[p] = {
                'text': generated_text,
                'lexical_diversity': metrics.lexical_diversity(generated_text),
                'bigram_overlap': metrics.n_gram_overlap(generated_text, n=2),
                'trigram_overlap': metrics.n_gram_overlap(generated_text, n=3),
                'spelling_accuracy': metrics.spell_check_percentage(generated_text),
                'repetition_ratio': metrics.repetition_ratio(generated_text)
            }
        
        results[model_name]['nucleus'] = nucleus_results
    
    return results

# Main function to run experiments
def main():
    # Set seeds for reproducibility
    set_seeds(42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    # Change this to your dataset path
    data_path = "shakespeare.txt"  # or "emily_dickinson.txt"
    
    # Load text
    text = load_data(data_path)
    print(f"Loaded {len(text)} characters from {data_path}")
    
    # Create data splits
    train_text, val_text, test_text = create_data_splits(text)
    print(f"Train: {len(train_text)} chars, Val: {len(val_text)} chars, Test: {len(test_text)} chars")
    
    # Create character mappings
    chars, char_to_idx, idx_to_char = create_char_mappings(text)
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size}")
    
    # Hyperparameters
    seq_length = 100
    batch_size = 64
    hidden_size = 256  # This is a starting point, we'll tune it
    num_epochs = 20
    
    # Prepare data for training
    train_x_batches, train_y_batches = prepare_data(train_text, char_to_idx, seq_length, batch_size)
    val_x_batches, val_y_batches = prepare_data(val_text, char_to_idx, seq_length, batch_size)
    test_x_batches, test_y_batches = prepare_data(test_text, char_to_idx, seq_length, batch_size)
    
    # Create data loaders
    train_loader = create_data_loader(train_x_batches, train_y_batches, batch_size, seq_length)
    val_loader = create_data_loader(val_x_batches, val_y_batches, batch_size, seq_length)
    test_loader = create_data_loader(test_x_batches, test_y_batches, batch_size, seq_length)
    
    # Define models
    rnn_model = VanillaRNN(vocab_size, hidden_size, vocab_size).to(device)
    lstm_1layer_model = LSTMModel(vocab_size, hidden_size, vocab_size, num_layers=1).to(device)
    lstm_2layer_model = LSTMModel(vocab_size, hidden_size, vocab_size, num_layers=2, dropout=0.2).to(device)
    
    # For C/D grade extension (optional)
    gru_model = GRUModel(vocab_size, hidden_size, vocab_size, num_layers=1).to(device)
    
    # Define optimizers
    rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)
    lstm_1layer_optimizer = optim.Adam(lstm_1layer_model.parameters(), lr=0.001)
    lstm_2layer_optimizer = optim.Adam(lstm_2layer_model.parameters(), lr=0.001)
    gru_optimizer = optim.Adam(gru_model.parameters(), lr=0.001)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create dictionary of models to train
    models = {
        'RNN': (rnn_model, rnn_optimizer),
        'LSTM_1Layer': (lstm_1layer_model, lstm_1layer_optimizer),
        'LSTM_2Layer': (lstm_2layer_model, lstm_2layer_optimizer),
        'GRU': (gru_model, gru_optimizer)
    }
    
    # Train each model
    for model_name, (model, optimizer) in models.items():
        print(f"\nTraining {model_name}...")
        
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train for one epoch
            train_loss = train_model(model, train_loader, optimizer, criterion, device)
            train_losses.append(train_loss)
            
            # Evaluate on validation set
            val_loss = evaluate_model(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            
            # Calculate perplexity
            train_perplexity = calculate_perplexity(train_loss)
            val_perplexity = calculate_perplexity(val_loss)
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch+1}/{num_epochs} | Time: {epoch_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f} | Train Perplexity: {train_perplexity:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Perplexity: {val_perplexity:.4f}")
        
        # Plot learning curves
        plot_learning_curves(train_losses, val_losses, model_name)
        
        # Save model
        torch.save(model.state_dict(), f"{model_name}_model.pt")
    
    # Evaluate on test set
    test_results = {}
    for model_name, (model, _) in models.items():
        test_loss = evaluate_model(model, test_loader, criterion, device)
        test_perplexity = calculate_perplexity(test_loss)
        test_results[model_name] = {
            'loss': test_loss,
            'perplexity': test_perplexity
        }
        print(f"{model_name} - Test Loss: {test_loss:.4f} | Test Perplexity: {test_perplexity:.4f}")
    
    # Generate text for qualitative comparison
    seed_text = "The king"  # Change as needed
    
    # Dictionary of trained models for text generation
    trained_models = {
        'RNN': rnn_model,
        'LSTM_1Layer': lstm_1layer_model,
        'LSTM_2Layer': lstm_2layer_model,
        'GRU': gru_model
    }
    
    # Evaluate text quality
    quality_results = evaluate_text_quality(
        trained_models, seed_text, char_to_idx, idx_to_char, 
        reference_text=train_text, device=device
    )
    
    # Print some sample generations
    for model_name, results in quality_results.items():
        print(f"\n--- {model_name} Generated Text ---")
        
        # Temperature = 1.0 sample
        print(f"Temperature = 1.0:")
        print(results['temperature'][1.0]['text'][:300])
        print(f"Lexical Diversity: {results['temperature'][1.0]['lexical_diversity']:.4f}")
        print(f"Bigram Overlap: {results['temperature'][1.0]['bigram_overlap']:.4f}")
        
        # Nucleus p = 0.9 sample
        print(f"\nNucleus p = 0.9:")
        print(results['nucleus'][0.9]['text'][:300])
        print(f"Lexical Diversity: {results['nucleus'][0.9]['lexical_diversity']:.4f}")
        print(f"Bigram Overlap: {results['nucleus'][0.9]['bigram_overlap']:.4f}")

if __name__ == "__main__":
    main()