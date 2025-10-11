from transformers import AutoTokenizer, AutoModel
import torch

model_name = "thenlper/gte-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# dummy input
text = ["Hello world"]
inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

# export
torch.onnx.export(
    model,
    (inputs['input_ids'], inputs['attention_mask']),
    "gte_base.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "last_hidden_state": {0: "batch", 1: "seq"}
    },
    opset_version=17
)