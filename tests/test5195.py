import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

model_path = "gte_base.onnx"
tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-base")

for text in ["I like machine learning", "I enjoy AI", "I love programming", "I like pizza"]:
    enc = tokenizer(text, add_special_tokens=True, return_attention_mask=True)
    print(text, enc["input_ids"], enc["attention_mask"])

# --- Example target and phrases ---
target = "I like machine learning"
phrases = [
    "I enjoy AI",
    "I love programming",
    "I hate homework",
    "I like pizza"
]

# --- Tokenize input for ONNX ---
def tokenize(text_list):
    enc = tokenizer(
        text_list,
        padding=True,
        truncation=True,
        return_tensors="np"
    )
    return enc['input_ids'], enc['attention_mask']

# --- Create ONNX session ---
session = ort.InferenceSession(model_path)

# Tokenize target
input_ids_target, attention_mask_target = tokenize([target])
inputs_target = {
    "input_ids": input_ids_target.astype(np.int64),
    "attention_mask": attention_mask_target.astype(np.int64)
}

# Tokenize phrases
input_ids_phrases, attention_mask_phrases = tokenize(phrases)
inputs_phrases = {
    "input_ids": input_ids_phrases.astype(np.int64),
    "attention_mask": attention_mask_phrases.astype(np.int64)
}

# --- Run ONNX model ---
target_output = session.run(["last_hidden_state"], inputs_target)[0]  # shape [1, seq_len, hidden_size]
phrases_output = session.run(["last_hidden_state"], inputs_phrases)[0]  # shape [N, seq_len, hidden_size]

# --- Mean pooling over tokens to get sentence embeddings ---
def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask[..., None]
    summed = (last_hidden_state * mask).sum(axis=1)
    counts = mask.sum(axis=1)
    return summed / counts

target_emb = mean_pooling(target_output, input_ids_target)
phrases_emb = mean_pooling(phrases_output, input_ids_phrases)

# --- Cosine similarities ---
def cosine_sim(a, b):
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.matmul(a_norm, b_norm.T)  # shape [1, N]

sims = cosine_sim(target_emb, phrases_emb)[0]
for phrase, sim in zip(phrases, sims):
    print(f"{phrase}: {sim:.4f}")