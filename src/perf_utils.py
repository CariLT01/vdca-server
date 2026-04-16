"""
Module containing utility classes and functions
for monitoring and improving the program's
performance.
"""

import time


class Utils:
    """
    Utility classes containing mostly functions for monitoring
    performance.
    """

    labels: dict[str, float] = {}
    n_labels_count: dict[str, int] = {}

    @staticmethod
    def tbegin(label: str):
        """
        Starts a timing with a label.
        Stands for `timing_begin`.
        **Make sure to end it with `tend`**.

        Args:
            label (str): label of the task
        """

        Utils.labels[label] = time.time()

        # check number of active tasks
        active_tasks_n = len(Utils.labels.keys())
        print(f"{'  ' * (active_tasks_n - 1)}> Executing task '{label}'...")

        Utils.n_labels_count[label] = active_tasks_n

    @staticmethod
    def tend(label: str, should_print: bool = True) -> float:
        """
        Ends a timing label started by `tbegin`.
        Stands for `timing_end`.

        Args:
            label (str): original label
            should_print (bool): whether or not to print timing in stdout

        Raises:
            RuntimeError: if a label was never created

        Returns:
            float: duration of task in milliseconds
        """

        if Utils.labels.get(label) == None:
            raise RuntimeError(f"Label '{label}' was never created!")

        start_time = Utils.labels[label]
        diff = time.time() - start_time

        ms = round(diff * 1000 * 100) / 100

        label_count = Utils.n_labels_count.get(label) or 1

        if should_print:
            print(f"{'  ' * (label_count - 1)}  Completed task '{label}' in {ms} ms")

        del Utils.labels[label]

        return ms
