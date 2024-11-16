import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import wraps
from typing import Callable, Generic, List, TypeVar

T = TypeVar("T")  # Input type
R = TypeVar("R")  # Result type


class ParallelProcessor(Generic[T, R]):
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or multiprocessing.cpu_count()

    def process_parallel(
        self, items: List[T], process_func: Callable[[T], R]
    ) -> List[R]:
        """
        Process items in parallel using a process pool

        Args:
            items: List of items to process
            process_func: Function to process each item

        Returns:
            List of processed results
        """
        results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Create futures with their indices
            futures = {
                executor.submit(process_func, item): i for i, item in enumerate(items)
            }
            # Initialize results list with None values
            results = [None] * len(items)
            # Fill results in correct positions as they complete
            for future in as_completed(futures):
                index = futures[future]
                results[index] = future.result()
        return results


def parallel_process(max_workers=None):
    """
    A decorator for parallel processing of iterables.

    Parameters:
    max_workers (int, optional): Maximum number of worker processes.
                               Defaults to number of CPU cores.

    Returns:
    function: Decorated function that processes items in parallel
    """
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    def decorator(func):
        @wraps(func)
        def wrapper(items, *args, **kwargs):
            if not items:
                return []

            results = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Create futures with their indices
                futures = {
                    executor.submit(func, item, *args, **kwargs): i
                    for i, item in enumerate(items)
                }
                # Initialize results list with None values
                results = [None] * len(items)
                # Fill results in correct positions as they complete
                for future in as_completed(futures):
                    index = futures[future]
                    results[index] = future.result()
                return results

        return wrapper

    return decorator
