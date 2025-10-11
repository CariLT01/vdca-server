from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name = "facebook/bart-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

sentence = "The walls from la Punta to the arsenal were protected by {} with parapets and a ditch."
choices = ['bulwarks', 'androids', 'batons', 'garnishes']

def score_choice(choice):
    prompt = sentence.format(choice)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.item()
    return -loss  # higher = better

scores = [(c, score_choice(c)) for c in choices]

# Step 1: find the minimum (most negative) value
min_score = min(score for _, score in scores)

# Step 2: shift all scores so the lowest becomes zero
shifted_scores = [(word, score - min_score) for word, score in scores]

# Step 3: sort descending (higher = better)
shifted_scores_sorted = sorted(shifted_scores, key=lambda x: x[1], reverse=True)

print(shifted_scores_sorted)

