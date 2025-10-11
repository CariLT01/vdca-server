from transformers import AutoTokenizer, AutoModelForMaskedLM
from pathlib import Path
import torch

model_name = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.eval()

# Example input
text = "The capital of Canada is [MASK]."
inputs = tokenizer(text, return_tensors="pt")

# Export to ONNX
torch.onnx.export(
    model,
    (inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]),
    Path("bert_mlm.onnx"),
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "token_type_ids": {0: "batch", 1: "seq"},
        "logits": {0: "batch", 1: "seq"}
    },
    opset_version=17
)