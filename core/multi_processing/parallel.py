import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Generic, List, TypeVar

T = TypeVar("T")  # Input type
R = TypeVar("R")  # Result type


class ParallelProcessor(Generic[T, R]):
    def __init__(self, max_workers: int = multiprocessing.cpu_count()):
        self.max_workers = max_workers

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


def parallel_process(max_workers: int = multiprocessing.cpu_count()):
    """
    Decorator for parallel processing of iterables.

    Args:
        max_workers: Maximum number of worker processes to use

    Returns:
        Decorated function that processes items in parallel
    """

    def decorator(func: Callable[[T], R]):
        def wrapper(items: List[T], *args, **kwargs) -> List[R]:
            results = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(func, item, *args, **kwargs) for item in items
                ]
                for future in as_completed(futures):
                    results.append(future.result())
            return results

        return wrapper

    return decorator
