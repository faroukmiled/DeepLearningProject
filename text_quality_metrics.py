import numpy as np
import torch
import re
from collections import Counter
import nltk
from nltk.util import ngrams
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import language_tool_python


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class TextQualityMetrics:
    def __init__(self, reference_text=None, vocab=None):
        self.reference_text = reference_text
        self.vocab = vocab
        self.tool = language_tool_python.LanguageTool('en-US')

        if reference_text:
            self.ref_tokens = reference_text.split()
            self.ref_bigrams = set(self._get_ngrams(reference_text, 2))
            self.ref_trigrams = set(self._get_ngrams(reference_text, 3))
            self.ref_4grams = set(self._get_ngrams(reference_text, 4))

    def _get_ngrams(self, text, n):
        tokens = text.split()
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def calculate_cross_entropy(self, model, text, device='cpu', skip_model_eval=False):
        if not hasattr(model, "__call__"):
            return None  # CharRNN is not a PyTorch model

        if not skip_model_eval and hasattr(model, 'eval'):
            model.eval()

        tokens = [self.vocab.index(char) for char in text if char in self.vocab]
        if len(tokens) < 2:
            return None
        input_tensor = torch.tensor(tokens[:-1], dtype=torch.long).unsqueeze(0).to(device)
        target_tensor = torch.tensor(tokens[1:], dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            logits = output[0].squeeze(0)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            loss = torch.nn.functional.nll_loss(log_probs, target_tensor.squeeze(0), reduction='mean')
        return loss.item()
    
    def calculate_perplexity(self, model, text, device='cpu', skip_model_eval=False):
        if not hasattr(model, "__call__"):
            return None  # CharRNN is not a PyTorch model

        if not skip_model_eval and hasattr(model, 'eval'):
            model.eval()

        tokens = [self.vocab.index(char) for char in text if char in self.vocab]
        tensor = torch.tensor(tokens, dtype=torch.long).to(device)
        with torch.no_grad():
            input_tensor = tensor[:-1].unsqueeze(0)
            target_tensor = tensor[1:].unsqueeze(0)
            outputs = model(input_tensor)
            log_probs = torch.nn.functional.log_softmax(outputs.squeeze(0), dim=1)
            nll = -log_probs.gather(1, target_tensor.transpose(0, 1)).squeeze(1)
            perplexity = torch.exp(torch.mean(nll)).item()
        return perplexity



    def grammar_errors(self, text):
        matches = self.tool.check(text)
        return len(matches)

    def n_gram_overlap(self, generated_text, n=2):
        if not self.reference_text:
            raise ValueError("Reference text must be provided to calculate n-gram overlap")
        gen_ngrams = set(self._get_ngrams(generated_text, n))
        if not gen_ngrams:
            return 0.0
        if n == 2:
            ref_ngrams = self.ref_bigrams
        elif n == 3:
            ref_ngrams = self.ref_trigrams
        elif n == 4:
            ref_ngrams = self.ref_4grams
        else:
            ref_ngrams = set(self._get_ngrams(self.reference_text, n))
        overlap = gen_ngrams.intersection(ref_ngrams)
        return len(overlap) / len(gen_ngrams)

    def lexical_diversity(self, text):
        words = text.split()
        return len(set(words)) / len(words) if words else 0

    def repetition_ratio(self, text, window_size=10):
        words = text.split()
        if len(words) <= window_size:
            return 0.0
        repetitions = 0
        total = 0
        for i in range(len(words) - window_size):
            window = tuple(words[i:i+window_size])
            next_word = words[i+window_size]
            for j in range(max(0, i-100), i):
                if tuple(words[j:j+window_size]) == window:
                    expected_next = words[j+window_size] if j+window_size < i else None
                    if expected_next == next_word:
                        repetitions += 1
                    break
            total += 1
        return repetitions / total if total > 0 else 0.0

    def spell_check_percentage(self, text):
        if not self.reference_text:
            raise ValueError("Reference text must be provided to calculate spelling correctness")
        known_words = set(word.lower() for word in self.reference_text.split())
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        if not words:
            return 0.0
        correct_words = sum(1 for word in words if word in known_words)
        return (correct_words / len(words)) * 100

    def calculate_bleu(self, generated_text, n_gram_weights=(0.25, 0.25, 0.25, 0.25)):
        if not self.reference_text:
            raise ValueError("Reference text must be provided to calculate BLEU score")
        reference_chunks = [self.reference_text.split()]
        generated_tokens = generated_text.split()
        smoothie = SmoothingFunction().method1
        return sentence_bleu(reference_chunks, generated_tokens, weights=n_gram_weights, smoothing_function=smoothie)

    def self_bleu(self, generated_texts):
        if len(generated_texts) <= 1:
            return 0.0
        scores = []
        smoothie = SmoothingFunction().method1
        for i, text in enumerate(generated_texts):
            references = [other.split() for j, other in enumerate(generated_texts) if j != i]
            hypothesis = text.split()
            score = sentence_bleu(references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
            scores.append(score)
        return sum(scores) / len(scores)

    def evaluate_temperature(self, model, seed_text, generate_func, temperatures=[0.7, 1.0, 1.3], max_length=200, skip_model_eval=False):
        results = {}
        for temp in temperatures:
            generated = generate_func(model, seed_text, temperature=temp, max_length=max_length)
            results[temp] = {
                "text": generated,
                "lexical_diversity": self.lexical_diversity(generated),
                "bigram_overlap": self.n_gram_overlap(generated, n=2),
                "trigram_overlap": self.n_gram_overlap(generated, n=3),
                "4gram_overlap": self.n_gram_overlap(generated, n=4),
                "spelling_accuracy": self.spell_check_percentage(generated),
                "repetition_ratio": self.repetition_ratio(generated),
                "bleu_score": self.calculate_bleu(generated),
                "cross_entropy": self.calculate_cross_entropy(model, generated, skip_model_eval=skip_model_eval),
                "grammar_errors": self.grammar_errors(generated)
            }
            try:
                results[temp]["perplexity"] = self.calculate_perplexity(model, generated, skip_model_eval=skip_model_eval)
            except:
                results[temp]["perplexity"] = None
        return results

    

    def compare_models(self, models_dict, seed_text, generate_func, temperature=1.0, max_length=200):
        results = {}
        for name, model in models_dict.items():
            generated = generate_func(model, seed_text, temperature=temperature, max_length=max_length)
            results[name] = {
                "text": generated,
                "lexical_diversity": self.lexical_diversity(generated),
                "bigram_overlap": self.n_gram_overlap(generated, n=2),
                "trigram_overlap": self.n_gram_overlap(generated, n=3),
                "spelling_accuracy": self.spell_check_percentage(generated),
                "repetition_ratio": self.repetition_ratio(generated),
                "bleu_score": self.calculate_bleu(generated),
                "cross_entropy": self.calculate_cross_entropy(model, generated),
                "grammar_errors": self.grammar_errors(generated)
            }
            try:
                results[name]["perplexity"] = self.calculate_perplexity(model, generated)
            except:
                results[name]["perplexity"] = None
        return results
