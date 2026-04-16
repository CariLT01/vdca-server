import time

class Utils:
    
    labels: dict[str, float] = {}
    n_labels_count: dict[str, int] = {}
    
    @staticmethod
    def tbegin(label: str):
        Utils.labels[label] = time.time()
        
        # check number of active tasks
        active_tasks_n = len(Utils.labels.keys())
        print(f"{'  ' * (active_tasks_n - 1)}> Executing task '{label}'...")
        
        Utils.n_labels_count[label] = active_tasks_n
    
    @staticmethod
    def tend(label: str, should_print: bool = True) -> int:
        
        if Utils.labels.get(label) == None:
            raise RuntimeError(f"Label '{label}' was never created!")
        
        start_time = Utils.labels[label]
        diff = time.time() - start_time
        
        ms = round(diff * 1000 * 100) / 100
        
        label_count = Utils.n_labels_count.get(label) or 1
        
        print(f"{'  ' * (label_count - 1)}  Completed task '{label}' in {ms} ms")
        
        del Utils.labels[label]
        
        return ms