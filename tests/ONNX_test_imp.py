import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

# --- Config ---
model_path = "bert_mlm.onnx"  # Path to your ONNX model
text = "The capital of Canada is [MASK]."

# --- Load tokenizer ---
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

# --- Encode text ---
inputs = tokenizer(text, return_tensors="np")  # returns numpy arrays
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
token_type_ids = inputs["token_type_ids"]

# --- Create ONNX Runtime session ---
session = ort.InferenceSession(model_path)

# --- Prepare input dict ---
onnx_inputs = {
    "input_ids": input_ids.astype(np.int64),
    "attention_mask": attention_mask.astype(np.int64),
    "token_type_ids": token_type_ids.astype(np.int64)
}

# --- Run inference ---
outputs = session.run(None, onnx_inputs)
logits = outputs[0]  # [batch_size, seq_len, vocab_size]

# --- Find mask token index ---
mask_token_id = tokenizer.mask_token_id  # should be 103
mask_index = np.where(input_ids == mask_token_id)[1][0]

# --- Pick predicted token ---
mask_logits = logits[0, mask_index, :]
predicted_id = int(mask_logits.argmax())
predicted_token = tokenizer.decode([predicted_id])

# --- Top 5 predictions ---
top5_ids = mask_logits.argsort()[-5:][::-1]
print(top5_ids)
top5_tokens = [tokenizer.decode([i]) for i in top5_ids]
top5_scores = mask_logits[top5_ids]

print(f"[MASK] predicted token: {predicted_token}")
print("Top 5 predictions at mask position:")
for t, s in zip(top5_tokens, top5_scores):
    print(f"  {t}: {s:.4f}")