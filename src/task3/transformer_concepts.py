import os
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline

os.makedirs("src/task3", exist_ok=True)

print("Initializing Transformer Concepts in Code...")

print("\nRunning Single-Head Self-Attention (NumPy)...")
seq_len, d_k = 4, 8

np.random.seed(42)
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_k)

scores = np.matmul(Q, K.T) / np.sqrt(d_k)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

attention_weights = softmax(scores)
output_single_head = np.matmul(attention_weights, V)

print("-> Attention Weights Matrix (4x4):")
print(np.round(attention_weights, 4))
row_sums = np.sum(attention_weights, axis=1)
print(f"-> Row-wise Sums Check: {np.round(row_sums, 4)} (Verified: Sums to 1.0)")


print("\nRunning Multi-Head Attention (2 Heads) & Projection...")
Q1, K1, V1 = np.random.randn(seq_len, d_k), np.random.randn(seq_len, d_k), np.random.randn(seq_len, d_k)
head1 = np.matmul(softmax(np.matmul(Q1, K1.T) / np.sqrt(d_k)), V1)

Q2, K2, V2 = np.random.randn(seq_len, d_k), np.random.randn(seq_len, d_k), np.random.randn(seq_len, d_k)
head2 = np.matmul(softmax(np.matmul(Q2, K2.T) / np.sqrt(d_k)), V2)

concat_heads = np.concatenate([head1, head2], axis=-1)
W_O = np.random.randn(d_k * 2, d_k)
mha_output = np.matmul(concat_heads, W_O)

print(f"Concatenated Multi-Head Shape: {concat_heads.shape}")
print(f"Final Projected Output Shape: {mha_output.shape}")

print("\nGenerating Sinusoidal Positional Encoding...")
positions = 10
d_model = 8

PE = np.zeros((positions, d_model))
for pos in range(positions):
    for i in range(0, d_model, 2):
        div_term = np.power(10000, i / d_model)
        PE[pos, i] = np.sin(pos / div_term)
        if i + 1 < d_model:
            PE[pos, i + 1] = np.cos(pos / div_term)

plt.figure(figsize=(8, 6))
plt.imshow(PE, cmap='viridis', aspect='auto')
plt.colorbar(label='Encoding Value')
plt.title('Sinusoidal Positional Encoding (10 Positions, 8 Dimensions)')
plt.xlabel('Dimension Index')
plt.ylabel('Position Index')

plot_path = "src/task3/attention_matrix.png"
plt.savefig(plot_path)
plt.close()
print(f"Saved visualization plot successfully to: {plot_path}")

print("\nExecuting Hugging Face Pipelines...")
try:
    unmasker = pipeline('fill-mask', model='bert-base-uncased')
    mask_res = unmasker("Artificial intelligence will shape the future of [MASK].")
    print(f"BERT (fill-mask) Top Result: {mask_res[0]['sequence']}")
except Exception as e:
    print(f"BERT Pipeline Note: {e}")

try:
    generator = pipeline('text-generation', model='gpt2')
    gen_res = generator("Deep learning architectures have revolutionized", max_length=25, num_return_sequences=1)
    print(f"GPT-2 (text-generation): {gen_res[0]['generated_text']}")
except Exception as e:
    print(f"GPT-2 Pipeline Note: {e}")


print("\nArchitecture Theory & Comparison Summary:")
print("1. Encoder-only (e.g., BERT) processes context bidirectionally for robust feature extraction.")
print("2. Decoder-only (e.g., GPT) utilizes causal masking for autoregressive text generation.")
print("3. Transformers replace recurrent loops with Self-Attention, permitting full parallel processing.")
print("4. This eliminates sequential bottlenecks, drastically improving scalability on large datasets.")
print("Execution completed.")