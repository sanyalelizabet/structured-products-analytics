"""Typed exceptions for data-fetching and market-data operations.

Callers can distinguish transport failures from missing data, and retry
logic can target only the transient (retryable) errors.

Hierarchy:
    DataFetchError                 -- base; always means "couldn't get the data"
      NetworkError                 -- transport-layer (timeout, conn reset, 5xx). Retryable.
        RateLimitError             -- HTTP 429. Retryable with backoff.
      DataUnavailableError         -- resource missing / empty / malformed (404, 4xx,
                                      empty response, bad JSON). Not retryable.
"""


class DataFetchError(Exception):
    """Base exception for data-fetching failures."""


class NetworkError(DataFetchError):
    """Transport-layer failure (timeout, connection error, 5xx). Retryable."""


class RateLimitError(NetworkError):
    """HTTP 429 — API rate limit exceeded. Retryable with backoff."""


class DataUnavailableError(DataFetchError):
    """Requested resource is missing, empty, or malformed. Not retryable."""
