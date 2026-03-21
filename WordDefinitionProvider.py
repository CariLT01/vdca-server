import requests


class WordDefinitionProvider:
    
    def __init__(self, api_url: str = "https://api.dictionaryapi.dev/api/v2/entries/en/{target}"):
        self.api_url = api_url
        
        
    def fetch_definition(self, target: str) -> str:
        if len(target.split(" ")) != 1: return None
        URL = self.api_url
        response = requests.get(URL)
        if (response.status_code != 200): return None
        
        # Fetch all meanings
        
        data = response.json()[0]
        
        index = -1
        maxDefinitionsLength = 0
        
        # Loop through meanings, found the most used indexes
        for i, meaning in enumerate(data["meanings"]):
            definitionsLength = len(meaning["definitions"])
            if definitionsLength > maxDefinitionsLength:
                index = i
        
        if index == -1: return None
        
        
        meanings = data["meanings"][index]
        definition = meanings["definitions"][0]["definition"]
        
        return definition