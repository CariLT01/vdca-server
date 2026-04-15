MODEL_NAME: str = "Qwen/Qwen3-0.6B"
from flask_socketio import SocketIO
from openai import OpenAI
from api_key import API_KEY, API_KEY_G4F
from g4f.client import Client
import multiprocessing


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
            api_key=API_KEY,
        )
        self.loaded = True
    
    @staticmethod
    def worker(queue, messages):
        try:
            
            g4f_client = Client(
                api_key=API_KEY_G4F,
                base_url="https://api.airforce/v1"
            )
            
            response = g4f_client.chat.completions.create(
                model="grok-4.1-fast",
                messages=messages
            )
            queue.put(response.choices[0].message.content)
        except Exception as e:
            queue.put(e)
        
    def run_with_timeout(self, messages, timeout=15):
        queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=self.worker, args=(queue, messages))
        p.start()
        p.join(timeout)

        if p.is_alive():
            print("Force killing g4f process")
            p.terminate()
            p.join()
            return None

        result = queue.get()

        if isinstance(result, Exception):
            raise result

        return result    
    
    
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
            print("Trying gpt4free")
        
            
            content_response = self.run_with_timeout(messages, timeout=10)
            if content_response is None:
                raise RuntimeError("g4f response timed out") from e
            print(f"g4f got response: {content_response}")

            return content_response