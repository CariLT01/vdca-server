MODEL_NAME: str = "Qwen/Qwen3-0.6B"
from flask_socketio import SocketIO
from openai import OpenAI
from API_Key import API_KEY, API_KEY_G4F
from g4f.client import Client


class LLMProvider:
    
    def __init__(self, socketIO_instance: SocketIO | None = None):
        
        print(f"-- Created LLM Provider instance")
        self.socket = socketIO_instance
        self.loaded = False
        
        self.loadProvider()
    
    def loadProvider(self):
        if self.loaded is True:
            print("warn: attempted to load provider again")
            return
        print("Created OpenRouter Client")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=API_KEY
        )
        self.g4f_client = Client(
            api_key=API_KEY_G4F,
            base_url="https://api.airforce/v1"
        )
        self.loaded = True
        
        
    
    def getResponse(self, prompt: str, stream_output=False):
        print(f"Prompt: {prompt}")
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # Full-response call (no streaming)
            response = self.client.chat.completions.create(
                model="nvidia/nemotron-3-super-120b-a12b:free",
                messages=messages,
                extra_body={"reasoning": {"enabled": True}}
            )
            
            # Extract the text from the first choice
            content_response = response.choices[0].message.content
            print(f"got response: {content_response}\n")
            
            return content_response
        except Exception as e:
            print(f"OpenRouter client failed with reason: {e}")
            print(f"Trying gpt4free")
            
            response = self.g4f_client.chat.completions.create(
                model="grok-4.1-fast",
                messages=messages
            )
            
            content_response = response.choices[0].message.content
            print(f"g4f got response: {content_response}")

            return content_response