"""HTTP download with retries for transient network errors (timeouts, connection drops)."""
import time
import requests


def download_with_retries(url, max_retries=5, timeout=120, backoff_base=2, **kwargs):
    """
    Download URL with retries for transient network errors.
    Retries on Timeout, ConnectTimeout, and ConnectionError.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=timeout, **kwargs)
            resp.raise_for_status()
            return resp
        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectTimeout,
            requests.exceptions.ConnectionError,
        ) as e:
            last_exc = e
            if attempt < max_retries - 1:
                wait_secs = backoff_base ** attempt
                time.sleep(wait_secs)
                continue
            raise last_exc
    raise last_exc
