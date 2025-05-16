import torch
import torch.nn.functional as F
import numpy as np
import random

def temperature_sampling(logits, temperature=1.0):
    """
    Sample from the output distribution using temperature.
    
    Args:
        logits (torch.Tensor): The logits output from the model
        temperature (float): Temperature parameter (higher = more random)
        
    Returns:
        int: The sampled token index
    """
    if temperature == 0:
        # Greedy sampling
        return torch.argmax(logits).item()
    
    # Apply temperature
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    
    # Sample from the distribution
    return torch.multinomial(probs, 1).item()

def nucleus_sampling(logits, p=0.9):
    """
    Sample from the top-p (nucleus) of the probability distribution.
    
    Args:
        logits (torch.Tensor): The logits output from the model
        p (float): The cumulative probability threshold (0.0 to 1.0)
        
    Returns:
        int: The sampled token index
    """
    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    
    # Calculate softmax probabilities
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    
    # Calculate cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Find indices where cumulative prob exceeds p
    sorted_indices_to_remove = cumulative_probs > p
    
    # Shift indices to remove first token after threshold
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = False
    
    # Create a mask for filtered logits
    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
    
    # Filter the logits and renormalize
    filtered_logits = logits.clone()
    filtered_logits[indices_to_remove] = float('-inf')
    probs = F.softmax(filtered_logits, dim=-1)
    
    # Sample from the filtered distribution
    return torch.multinomial(probs, 1).item()

def generate_text_with_temperature(model, seed_text, char_to_idx, idx_to_char, 
                                  max_length=500, temperature=1.0, device='cpu'):
    """
    Generate text using temperature sampling.
    
    Args:
        model: The language model
        seed_text (str): Initial text to start generation
        char_to_idx (dict): Mapping from characters to indices
        idx_to_char (dict): Mapping from indices to characters
        max_length (int): Maximum length of generated text
        temperature (float): Temperature parameter 
        device (str): Device to run model on ('cpu' or 'cuda')
        
    Returns:
        str: The generated text
    """
    model.eval()
    generated_text = seed_text
    
    # Convert seed to tensor
    input_sequence = [char_to_idx.get(char, 0) for char in seed_text]
    input_tensor = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0).to(device)
    
    # Initialize hidden state (implementation depends on your model)
    hidden = None  # Your model might need initialization
    
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            outputs, hidden = model(input_tensor, hidden)
            
            # Get prediction for next character
            next_char_logits = outputs[0, -1, :]
            
            # Sample next character
            next_char_idx = temperature_sampling(next_char_logits, temperature)
            next_char = idx_to_char[next_char_idx]
            
            # Add to generated text
            generated_text += next_char
            
            # Update input tensor for next iteration
            input_tensor = torch.tensor([[next_char_idx]], dtype=torch.long).to(device)
    
    return generated_text

def generate_text_with_nucleus(model, seed_text, char_to_idx, idx_to_char, 
                              max_length=500, p=0.9, device='cpu'):
    """
    Generate text using nucleus (top-p) sampling.
    
    Args:
        model: The language model
        seed_text (str): Initial text to start generation
        char_to_idx (dict): Mapping from characters to indices
        idx_to_char (dict): Mapping from indices to characters
        max_length (int): Maximum length of generated text
        p (float): The nucleus probability threshold
        device (str): Device to run model on ('cpu' or 'cuda')
        
    Returns:
        str: The generated text
    """
    model.eval()
    generated_text = seed_text
    
    # Convert seed to tensor
    input_sequence = [char_to_idx.get(char, 0) for char in seed_text]
    input_tensor = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0).to(device)
    
    # Initialize hidden state (implementation depends on your model)
    hidden = None  # Your model might need initialization
    
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            outputs, hidden = model(input_tensor, hidden)
            
            # Get prediction for next character
            next_char_logits = outputs[0, -1, :]
            
            # Sample next character
            next_char_idx = nucleus_sampling(next_char_logits, p)
            next_char = idx_to_char[next_char_idx]
            
            # Add to generated text
            generated_text += next_char
            
            # Update input tensor for next iteration
            input_tensor = torch.tensor([[next_char_idx]], dtype=torch.long).to(device)
    
    return generated_text

def compare_sampling_methods(model, seed_text, char_to_idx, idx_to_char, 
                            temperatures=[0.5, 0.7, 1.0, 1.3], 
                            p_values=[0.5, 0.7, 0.9, 0.95],
                            max_length=200, device='cpu'):
    """
    Compare different sampling methods (temperature and nucleus).
    
    Args:
        model: The language model
        seed_text (str): Initial text to start generation
        char_to_idx (dict): Mapping from characters to indices
        idx_to_char (dict): Mapping from indices to characters
        temperatures (list): List of temperature values to try
        p_values (list): List of nucleus p values to try
        max_length (int): Maximum length of generated text
        device (str): Device to run model on ('cpu' or 'cuda')
        
    Returns:
        dict: Dictionary of generated text with different sampling methods
    """
    results = {
        'temperature': {},
        'nucleus': {}
    }
    
    # Temperature sampling
    for temp in temperatures:
        results['temperature'][temp] = generate_text_with_temperature(
            model, seed_text, char_to_idx, idx_to_char, 
            max_length=max_length, temperature=temp, device=device
        )
    
    # Nucleus sampling
    for p in p_values:
        results['nucleus'][p] = generate_text_with_nucleus(
            model, seed_text, char_to_idx, idx_to_char, 
            max_length=max_length, p=p, device=device
        )
    
    return results