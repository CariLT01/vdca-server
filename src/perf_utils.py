import time

class Utils:
    
    labels: dict[str, int] = {}
    
    @staticmethod
    def begin_time(label: str):
        Utils.labels[label] = time.time()
    
    @staticmethod
    def end_time(label: str, should_print: bool = True) -> int:
        
        if Utils.labels.get(label) == None:
            raise RuntimeError(f"Label '{label}' was never created!")
        
        start_time = Utils.labels[label]
        diff = time.time() - start_time
        
        ms = round(diff * 1000 * 100) / 100
        
        print(f"Task '{label}' took approximately {ms} milliseconds to complete")
        
        return ms