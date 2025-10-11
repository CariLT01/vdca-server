from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from transformers.onnx import export
from sentence_transformers import SentenceTransformer
import torch

model_name = "thenlper/gte-base"
model = SentenceTransformer(model_name)

# Paths
output_dir = Path("gte-base-onnx")
output_dir.mkdir(exist_ok=True)

# Load underlying model
hf_model_name = model._first_module().auto_model_name
hf_model = AutoModel.from_pretrained(hf_model_name)
tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)
sample_text = ["This is a test sentence."]
inputs = tokenizer(sample_text, return_tensors="pt")


torch.onnx.export(
    hf_model,                    # model to export
    (inputs['input_ids'], inputs['attention_mask']),  # model input tuple
    "gte_base.onnx",              # output ONNX file
    input_names=['input_ids', 'attention_mask'],
    output_names=['output'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'seq_len'},
        'attention_mask': {0: 'batch_size', 1: 'seq_len'},
        'output': {0: 'batch_size'}
    },
    opset_version=17
)