import os as os_module
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from text_quality_metrics import TextQualityMetrics

# -------------------------
plots_dir = os_module.path.join(os_module.path.dirname(__file__), "plots")
os_module.makedirs(plots_dir, exist_ok=True)
book_path = "shakespeare.txt"

# -------------------------
def read_book(file_path):
    with open(file_path, 'r') as f:
        return f.read()

def plot_loss(losses):
    iterations, values = zip(*losses)
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, values)
    plt.xlabel('Iteration')
    plt.ylabel('Smooth Loss')
    plt.title('RNN Training Progress')
    plt.grid(True)
    plt.savefig(os_module.path.join(plots_dir, 'loss_plot.png'))
    plt.show()

def one_hot_encode(idx, size):
    vec = np.zeros((size, 1))
    vec[idx] = 1
    return vec

def rel_error(x, y):
    return np.max(np.abs(x - y) / np.maximum(1e-8, np.abs(x) + np.abs(y)))


# -------------------------
# CharRNN Class
# -------------------------

class CharRNN:
    def __init__(self, data, hidden_size=100, seq_length=25, learning_rate=0.001):
        self.data = data
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate

        self.chars = list(set(data))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

        self.rng = np.random.default_rng(42)
        self.RNN = {}
        self.initialize_parameters()

        self.m = {k: np.zeros_like(v) for k, v in self.RNN.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.RNN.items()}
        self.beta1, self.beta2, self.epsilon = 0.9, 0.999, 1e-8
        self.t = 0

        self.e = 0
        self.epoch = 0
        self.smooth_loss = -np.log(1.0/self.vocab_size) * self.seq_length

    def save_model(self, path):
        np.savez(path,
             **self.RNN,
             t=self.t,
             **{f"m_{k}": v for k, v in self.m.items()},
             **{f"v_{k}": v for k, v in self.v.items()})

    def load_model(self, path):
        checkpoint = np.load(path)
        self.RNN = {k: checkpoint[k] for k in ['U', 'W', 'V', 'b', 'c']}
        self.m = {k: checkpoint[f"m_{k}"] for k in self.RNN}
        self.v = {k: checkpoint[f"v_{k}"] for k in self.RNN}
        self.t = int(checkpoint['t'])
        print("Type of os:", type(os_module))

    def initialize_parameters(self):
        H, V = self.hidden_size, self.vocab_size
        self.RNN['U'] = (1/np.sqrt(2*V)) * self.rng.standard_normal((H, V))
        self.RNN['W'] = (1/np.sqrt(2*H)) * self.rng.standard_normal((H, H))
        self.RNN['V'] = (1/np.sqrt(H)) * self.rng.standard_normal((V, H))
        self.RNN['b'] = np.zeros((H, 1))
        self.RNN['c'] = np.zeros((V, 1))

    def forward_pass(self, X, Y, h_prev):
        xs, hs, as_, os, ps = {}, {}, {}, {}, {}
        hs[-1] = h_prev.copy()
        loss = 0

        for t in range(len(X)):
            xs[t] = one_hot_encode(X[t], self.vocab_size)
            as_[t] = np.dot(self.RNN['W'], hs[t-1]) + np.dot(self.RNN['U'], xs[t]) + self.RNN['b']
            hs[t] = np.tanh(as_[t])
            os[t] = np.dot(self.RNN['V'], hs[t]) + self.RNN['c']
            e = np.exp(os[t] - np.max(os[t]))
            ps[t] = e / np.sum(e, axis=0)
            loss += -np.log(ps[t][Y[t], 0])

        loss /= len(X)
        return xs, hs, as_, os, ps, loss, hs[len(X)-1]

    def backward_pass(self, X, Y, xs, hs, as_, ps):
        dRNN = {k: np.zeros_like(v) for k, v in self.RNN.items()}
        dhnext = np.zeros((self.hidden_size, 1))

        for t in reversed(range(len(X))):
            dy = ps[t].copy()
            dy[Y[t]] -= 1
            dy /= len(X)

            dRNN['V'] += np.dot(dy, hs[t].T)
            dRNN['c'] += dy

            dh = np.dot(self.RNN['V'].T, dy) + dhnext
            da = (1 - hs[t]**2) * dh

            dRNN['b'] += da
            dRNN['W'] += np.dot(da, hs[t-1].T)
            dRNN['U'] += np.dot(da, xs[t].T)
            dhnext = np.dot(self.RNN['W'].T, da)

        for k in dRNN:
            dRNN[k] = np.clip(dRNN[k], -5, 5)
        return dRNN

    def adam_update(self, grads):
        self.t += 1
        for k in self.RNN:
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * grads[k]**2
            m_hat = self.m[k] / (1 - self.beta1**self.t)
            v_hat = self.v[k] / (1 - self.beta2**self.t)
            self.RNN[k] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def sample(self, h, x_idx, n):
        h_prev = h.copy()
        sampled_indices = []
        for _ in range(n):
            x = one_hot_encode(x_idx, self.vocab_size)
            a = np.dot(self.RNN['W'], h_prev) + np.dot(self.RNN['U'], x) + self.RNN['b']
            h_prev = np.tanh(a)
            o = np.dot(self.RNN['V'], h_prev) + self.RNN['c']
            p = np.exp(o - np.max(o)) / np.sum(np.exp(o - np.max(o)))
            cp = np.cumsum(p)
            a = self.rng.uniform()
            x_idx = np.argmax(cp >= a)
            sampled_indices.append(x_idx)
        return ''.join(self.idx_to_char[i] for i in sampled_indices)

    def train(self, num_iterations=10000, print_every=100, sample_every=1000):
        h_prev = np.zeros((self.hidden_size, 1))
        losses, sample_texts = [], []
        iteration = 0
        gamma = 0.9

        while self.epoch < 3:
            if self.e + self.seq_length + 1 >= len(self.data):
                h_prev = np.zeros((self.hidden_size, 1))
                self.e = 0
                self.epoch += 1
                print(f"Starting epoch {self.epoch}")

            X = [self.char_to_idx[ch] for ch in self.data[self.e:self.e + self.seq_length]]
            Y = [self.char_to_idx[ch] for ch in self.data[self.e + 1:self.e + self.seq_length + 1]]

            xs, hs, as_, os, ps, loss, h_prev = self.forward_pass(X, Y, h_prev)
            grads = self.backward_pass(X, Y, xs, hs, as_, ps)
            self.adam_update(grads)

            if iteration == 0:
                self.smooth_loss = loss
            else:
                self.smooth_loss = gamma * self.smooth_loss + (1 - gamma) * loss

            if iteration % print_every == 0:
                print(f"iter = {iteration:6}, epoch = {self.epoch}, smooth loss = {self.smooth_loss:.5f}")
                losses.append((iteration, self.smooth_loss))

            if iteration % sample_every == 0:
                sample_text = self.sample(h_prev, X[0], 200)
                print(f"{'-'*40}\nSample at iter {iteration}:\n{sample_text}\n{'-'*40}")
                sample_path = os_module.path.join(plots_dir, f'sample_iter_{iteration}.txt')
                with open(sample_path, 'w') as f:
                    f.write(sample_text)
                sample_texts.append((iteration, sample_text))

            if iteration % 10000 == 0:
                checkpoint_path = os_module.path.join(plots_dir, f'model_checkpoint_{iteration}.npz')
                self.save_model(checkpoint_path)
                print(f"Checkpoint saved at iteration {iteration}")

            self.e += self.seq_length
            iteration += 1

        final_model_path = os_module.path.join(plots_dir, 'final_model.npz')
        self.save_model(final_model_path)
        print("Final model saved.")

        return losses, sample_texts
  


def plot_enhanced_loss(loss_data, eta=0.001, seq_length=25, beta1=0.9, beta2=0.999, save_path="plots/enhanced_loss_plot.png"):
    import matplotlib.pyplot as plt

    iterations, losses = zip(*loss_data)
    target_iters = [100, 500, 1000, 5000, 10000, 20000, 30000,
                    40000, 50000, 60000, 70000, 80000, 90000, 100000]

    print("Loss values at specific iterations:")
    loss_dict = dict(loss_data)
    for ti in target_iters:
        if ti in loss_dict:
            print(f"Iteration {ti}: Loss = {loss_dict[ti]:.6f}")
        else:
            print(f"Iteration {ti}: Not recorded")

    best_loss = min(losses)
    best_iter = iterations[losses.index(best_loss)]

    plt.figure(figsize=(10, 5))
    plt.plot(iterations, losses, label="Smooth Loss")
    plt.scatter(best_iter, best_loss, color='red')
    plt.text(best_iter, best_loss + 1.5, f"Best: {best_loss:.2f} @ {best_iter}", fontsize=9)
    plt.title(f"Loss Curve (η={eta}, seq_len={seq_length}, β₁={beta1}, β₂={beta2})")
    plt.xlabel("Iteration")
    plt.ylabel("Smooth Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()



# -------------------------
# Gradient Checking
# -------------------------
def check_gradients():
    np.random.seed(42)
    torch.manual_seed(42)

    hidden_size = 10
    seq_length = 25
    vocab_size = 27

    # Generate random input/output sequences
    X = np.random.randint(0, vocab_size, seq_length)
    Y = np.random.randint(0, vocab_size, seq_length)
    h0 = np.zeros((hidden_size, 1))

    # Instantiate CharRNN model
    model = CharRNN(''.join(chr(65 + i % vocab_size) for i in range(100)), hidden_size, seq_length)
    xs, hs, as_, os, ps, loss, _ = model.forward_pass(X, Y, h0)
    grads = model.backward_pass(X, Y, xs, hs, as_, ps)

    # Convert to double precision tensors for better precision in gradients
    U = torch.tensor(model.RNN['U'], requires_grad=True, dtype=torch.float64)
    W = torch.tensor(model.RNN['W'], requires_grad=True, dtype=torch.float64)
    V = torch.tensor(model.RNN['V'], requires_grad=True, dtype=torch.float64)
    b = torch.tensor(model.RNN['b'], requires_grad=True, dtype=torch.float64)
    c = torch.tensor(model.RNN['c'], requires_grad=True, dtype=torch.float64)
    h_prev = torch.zeros((hidden_size, 1), requires_grad=True, dtype=torch.float64)

    loss_pt = 0
    hs_pt = {-1: h_prev}

    for t in range(seq_length):
        x = torch.zeros(vocab_size, 1, dtype=torch.float64)
        x[X[t]] = 1.0
        a = W @ hs_pt[t - 1] + U @ x + b
        h = torch.tanh(a)
        o = V @ h + c
        log_p = F.log_softmax(o, dim=0)  # numerically stable log-softmax
        loss_pt += -log_p[Y[t]]
        hs_pt[t] = h

    loss_pt /= seq_length
    loss_pt.backward()

    # Report relative errors
    print("Gradient check (relative errors should be < 1e-5 for correctness):")
    print(f"Rel error U: {rel_error(U.grad.numpy(), grads['U'])}")
    print(f"Rel error W: {rel_error(W.grad.numpy(), grads['W'])}")
    print(f"Rel error V: {rel_error(V.grad.numpy(), grads['V'])}")
    print(f"Rel error b: {rel_error(b.grad.numpy(), grads['b'])}")
    print(f"Rel error c: {rel_error(c.grad.numpy(), grads['c'])}")


def generate_func(model, seed_text, temperature=1.0, max_length=200):
    h = np.zeros((model.hidden_size, 1))
    idx = model.char_to_idx.get(seed_text[0], 0)
    return model.sample(h, idx, max_length)

# -------------------------
def main():
    global plots_dir

    book_data = read_book(book_path)
    print("Loaded dataset from:", book_path)

    model = CharRNN(book_data)

    print("Performing gradient check before training...")
    check_gradients()

    checkpoint_path = os_module.path.join(plots_dir, 'final_model.npz')
    if os_module.path.exists(checkpoint_path):
        print("Loading saved model...")
        model.load_model(checkpoint_path)

    print("Starting training...")
    losses, samples = model.train(num_iterations=100000, print_every=100, sample_every=10000)

    np.save(os_module.path.join(plots_dir, 'smooth_loss.npy'), np.array(losses))
    plot_enhanced_loss(
        losses,
        eta=model.learning_rate,
        seq_length=model.seq_length,
        beta1=model.beta1,
        beta2=model.beta2
    )

    print("\nFinal sample (1000 characters):")
    final_sample = model.sample(np.zeros((model.hidden_size, 1)), model.char_to_idx[model.data[0]], 1000)
    print(final_sample)

    with open(os_module.path.join(plots_dir, "final_sample_1000_chars.txt"), "w") as f:
        f.write(final_sample)

    model.save_model(os_module.path.join(plots_dir, 'final_model.npz'))

    # -------------------------
    # Evaluation with Metrics
    # -------------------------
    metrics = TextQualityMetrics(reference_text=book_data, vocab=model.chars)

    results = metrics.evaluate_temperature(
        model=model,
        seed_text="To be",
        generate_func=generate_func,
        temperatures=[0.7, 1.0, 1.3],
        max_length=300,
        skip_model_eval=True
    )

    print("\n--- Evaluation Results ---")
    for temp, res in results.items():
        print(f"\nTemp = {temp}")
        print(f"BLEU: {res['bleu_score']:.4f}")
        print(f"Perplexity: {res['perplexity']}")
        print(f"Grammar Errors: {res['grammar_errors']}")
        print(f"Cross-Entropy: {res['cross_entropy']:.4f}")
        print(f"Sample Text:\n{res['text'][:300]}...\n")

    results_path = os_module.path.join(plots_dir, "temperature_eval.txt")
    with open(results_path, "w") as f:
        f.write("--- Evaluation Results ---\n")
        for temp, res in results.items():
            f.write(f"\nTemp = {temp}\n")
            f.write(f"BLEU: {res['bleu_score']:.4f}\n")
            f.write(f"Perplexity: {res['perplexity']}\n")
            f.write(f"Grammar Errors: {res['grammar_errors']}\n")
            f.write(f"Cross-Entropy: {res['cross_entropy']:.4f}\n")
            f.write(f"Sample Text:\n{res['text'][:300]}...\n")

    temps = list(results.keys())
    bleu_scores = [res['bleu_score'] for res in results.values()]
    perplexities = [res['perplexity'] for res in results.values()]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(temps, bleu_scores, marker='o')
    plt.title("BLEU Score vs Temperature")
    plt.xlabel("Temperature")
    plt.ylabel("BLEU Score")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(temps, perplexities, marker='o', color='orange')
    plt.title("Perplexity vs Temperature")
    plt.xlabel("Temperature")
    plt.ylabel("Perplexity")
    plt.grid(True)

    plt.tight_layout()
    plot_path = os_module.path.join(plots_dir, "temperature_metrics.png")
    plt.savefig(plot_path)
    plt.show()

    print("\n--- Evaluation of Final Sample vs Original Text ---")

    segment_length = 100
    num_segments = 3
    offsets = [i * segment_length for i in range(num_segments)]
    scores_list = []

    for offset in offsets:
        ref_excerpt = book_data[offset:offset+segment_length]
        gen_excerpt = final_sample[offset:offset+segment_length]

        metrics_segment = TextQualityMetrics(reference_text=ref_excerpt, vocab=model.chars)
        scores = {
            "lexical_diversity": metrics_segment.lexical_diversity(gen_excerpt),
            "bigram_overlap": metrics_segment.n_gram_overlap(gen_excerpt, n=2),
            "trigram_overlap": metrics_segment.n_gram_overlap(gen_excerpt, n=3),
            "4gram_overlap": metrics_segment.n_gram_overlap(gen_excerpt, n=4),
            "spelling_accuracy": metrics_segment.spell_check_percentage(gen_excerpt),
            "repetition_ratio": metrics_segment.repetition_ratio(gen_excerpt),
            "bleu_score": metrics_segment.calculate_bleu(gen_excerpt),
            "grammar_errors": metrics_segment.grammar_errors(gen_excerpt),
            "cross_entropy": metrics_segment.calculate_cross_entropy(model, gen_excerpt),
            "perplexity": metrics_segment.calculate_perplexity(model, gen_excerpt),
        }
        scores_list.append(scores)

    avg_scores = {k: np.mean([score[k] for score in scores_list]) for k in scores_list[0]}

    for k, v in avg_scores.items():
        print(f"{k.replace('_', ' ').capitalize()}: {v}")

    with open(os_module.path.join(plots_dir, "final_sample_evaluation.txt"), "w") as f:
        f.write("--- Final Sample Evaluation (Avg of 3x100-char segments) ---\n")
        for k, v in avg_scores.items():
            f.write(f"{k.replace('_', ' ').capitalize()}: {v}\n")

    check_gradients()
    print("Gradient checking completed.")

if __name__ == "__main__":
    main()



# The code above implements a character-level RNN for text generation using numpy.
# It includes functions for reading text, plotting loss, one-hot encoding, and calculating relative error.
# The CharRNN class contains methods for initializing parameters, forward and backward passes,
# Adam optimization, sampling characters, and training the model.
# The script also includes a gradient checking function to verify the correctness of the gradients.
# The main function orchestrates the training process and generates final samples.
# The code is structured to be modular and reusable, allowing for easy modifications and enhancements.
# The training process is designed to run for a specified number of iterations,
# with periodic loss reporting and character sampling.
# The final sample is saved to a text file for further analysis.
# The code is designed to be run as a standalone script, with the option to check gradients.
# The gradient checking function compares the gradients computed by the RNN implementation
# with those computed by PyTorch, ensuring the correctness of the implementation.
# The code is well-documented, with clear function and class definitions,
# making it easy to understand and follow.
# The implementation is efficient and leverages numpy for numerical computations,
# ensuring fast execution times.
# The use of Adam optimization helps in faster convergence during training,
# making the model more robust to different learning rates.     

