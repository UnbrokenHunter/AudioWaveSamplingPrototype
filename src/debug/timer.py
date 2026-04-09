import time
from functools import wraps

def time_method(threshold=0.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            elapsed = end - start

            if elapsed >= threshold:
                print(f"{func.__name__} took {elapsed:.6f} seconds")

            return result
        return wrapper
    return decorator