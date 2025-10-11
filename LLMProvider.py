MODEL_NAME: str = "Qwen/Qwen3-0.6B"

import torch
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from flask_socketio import SocketIO, emit

class LLMProvider:
    
    def __init__(self, socketIO_instance: SocketIO | None = None):
        
        print(f"-- Created LLM Provider instance")
        self.socket = socketIO_instance
    
    def loadProvider(self):
        
        print(f"Loading LLM model {MODEL_NAME}...")
        print(f"Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        print(f"Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
        print(f"Compiling model for better performance...")
        self.model = torch.compile(self.model, backend="inductor")
        
        num_threads = int(torch.get_num_threads())
        print(f"-- Using {num_threads} for LLM inference")
        
        torch.set_num_threads(num_threads)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        print(f"-- Using {self.device.upper()} device")
    
    def _gen(self):
        with torch.inference_mode():
            
            self.model.generate(
                input_ids=self.model_inputs.input_ids,
                attention_mask=self.model_inputs.attention_mask,
                max_new_tokens=2_147_483_647,
                do_sample=True,        # enable stochastic generation
                temperature=0.3,       # moderate randomness for creative thinking
                top_p=0.95,            # nucleus sampling: keep cumulative probability <= 0.95
                top_k=20,              # only consider top 20 tokens at each step
                streamer=self.streamer,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            ) 
        
    
    def getResponse(self, prompt: str, stream_output=True):
        
        messages = [{"role": "user", "content": prompt}]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        
        self.model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True, timeout=10.0, skip_prompt=True)
        
        # Generate response
        
        thread = threading.Thread(target=self._gen)
        thread.start()
        
        content_response=""
        seen_delim = False
        think_delim = "</think>"
        
        for chunk in self.streamer:
            if self.socket:
                self.socket.emit("log", chunk)
            if not seen_delim:
                if think_delim in chunk:
                    before, after = chunk.split(think_delim)
                    if stream_output:
                        print(before, end="", flush=True)
                        print("\ncontent:", end=" ", flush=True)
                        if after and stream_output:
                            print(after, end="", flush=True)
                    content_response += after  # store final content after the think tag
                    seen_delim = True
                else:
                    if stream_output:
                        print(chunk, end="", flush=True)
            else:
                if stream_output:
                    print(chunk, end="", flush=True)
                content_response += chunk  # accumulate remaining content
        
        thread.join()
        print("")
        
        return content_response
        
        