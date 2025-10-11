from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

print(f"Loading model...")

model_name = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
modelRoberta = AutoModelForMaskedLM.from_pretrained(model_name)

def rank_choices(sentence, choices):
    """
    Return MLM scores for candidate words in the given sentence.
    Returns a list of floats corresponding to `choices` (unsorted).
    """
    mask_token = tokenizer.mask_token
    sentence = sentence.replace("[MASK]", mask_token)
    input_ids = tokenizer(sentence, return_tensors="pt")["input_ids"]
    mask_idx = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    with torch.no_grad():
        outputs = modelRoberta(input_ids)
        logits = outputs.logits[0, mask_idx, :]  # shape: [mask_position, vocab_size]

    scores = []
    for word in choices:
        token_ids = tokenizer(word, add_special_tokens=False)["input_ids"]
        score = logits[:, token_ids].mean().item()  # average if multiple tokens
        scores.append(score)

    return scores


print(f"Compute...")

# Example usage
sentence = "The strangeness of entering as [MASK] the house where she had so long commanded."
choices = ["an investor", "a suppliant", "an authority", "a fanatic"]

ranked = rank_choices(sentence, choices)
for score in ranked:
    print(score)