import torch
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np

# --- Load tokenizer ---
model_name = "thenlper/gte-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# --- Load ONNX model ---
ort_session = ort.InferenceSession("gte-base.onnx")

# --- Prepare input ---
text = "The capital of Canada is [MASK]."
inputs = tokenizer(text, return_tensors="np")  # numpy arrays for ONNX Runtime

# --- ONNX Runtime input dict ---
ort_inputs = {
    "input_ids": inputs["input_ids"],
    "attention_mask": inputs["attention_mask"]
}

# --- Run inference ---
ort_outs = ort_session.run(None, ort_inputs)
logits = ort_outs[0]  # shape: [batch_size, seq_len, vocab_size]

# --- Find [MASK] position ---
mask_token_id = tokenizer.mask_token_id
mask_index = np.where(inputs["input_ids"][0] == mask_token_id)[0][0]

# --- Pick top predictions at mask position ---
top_k = 5
mask_logits = logits[0, mask_index]
top_ids = mask_logits.argsort()[-top_k:][::-1]

print("Top predictions for [MASK]:")
for idx in top_ids:
    token = tokenizer.decode([idx])
    score = mask_logits[idx]
    print(f"{token}: {score:.4f}")