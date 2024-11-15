import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
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
            futures = [executor.submit(process_func, item) for item in items]
            for future in as_completed(futures):
                results.append(future.result())
        return results
