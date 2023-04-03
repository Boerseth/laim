import os
import requests
from typing import Any, Callable


def str_file_cache(filename: str) -> Callable[[Callable[..., str]], Callable[..., str]]:
    def decorator(inner: Callable[..., str]) -> Callable[..., str]:
        def outer(*args: Any, **kwargs: Any) -> str:
            if os.path.exists(filename) and not kwargs.get("force_refresh"):
                with open(filename, "r") as f:
                    return f.read()
            result = inner(*args, **kwargs)
            assert isinstance(result, str)
            with open(filename, "w") as f:
                f.write(result)
            return result

        return outer

    return decorator


def get_http(url: str) -> str:
    return requests.get(url).text
