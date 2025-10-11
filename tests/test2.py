from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

model_name = "thenlper/gte-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

sentence = "But when I visited there were no queues in the waiting lounge."
target_word = "queues"
choices = ["complaints", "chairs", "workers", "lines"]

# Tokenize sentence
inputs = tokenizer(sentence, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

last_hidden = outputs.last_hidden_state  # [1, seq_len, hidden_size]

# Find token indices for target_word
target_tokens = tokenizer(target_word, add_special_tokens=False)["input_ids"]
input_ids = inputs["input_ids"][0].tolist()

# Find starting index
start_idx = None
for i in range(len(input_ids) - len(target_tokens) + 1):
    if input_ids[i:i+len(target_tokens)] == target_tokens:
        start_idx = i
        break

assert start_idx is not None, "Target word not found!"

# Average embedding of target tokens
target_emb = last_hidden[0, start_idx:start_idx+len(target_tokens), :].mean(dim=0, keepdim=True)

# Encode choices and compute similarity
for choice in choices:
    choice_tokens = tokenizer(choice, add_special_tokens=False)["input_ids"]
    choice_emb = model(torch.tensor([choice_tokens]))[0].mean(dim=1)  # average
    sim = F.cosine_similarity(target_emb, choice_emb)
    print(f"{choice}: {sim.item():.4f}")