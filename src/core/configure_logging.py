"""
Central logging configuration for the application.

Sets root logging to DEBUG and forces verbose libraries (HTTP clients,
HF/transformers, litellm, LangChain) down to DEBUG explicitly, since
several of them set their own logger level higher than root by default
and won't emit debug output from basicConfig alone.

Also enables raw HTTP connection-level debug output (DNS, connect, TLS
handshake) via http.client, which is the clearest signal when diagnosing
a hung or slow network call.

Call configure_logging() once, as early as possible in the application's
startup path (before any other library does its own logging setup),
so this configuration takes effect before those libraries initialize
their loggers.
"""

import logging
import sys
import http.client as http_client

_NOISY_LOGGERS = [
    "urllib3", "requests", "httpx", "httpcore",
    "sentence_transformers", "transformers", "huggingface_hub",
    "litellm", "langchain", "langchain_community",
]


def configure_logging(enabled: bool = True, level: int = logging.DEBUG, http_debug: bool = False) -> None:

    if not enabled:
        http_client.HTTPConnection.debuglevel = 0
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )

    for logger_name in _NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(level)

    http_client.HTTPConnection.debuglevel = 1 if http_debug else 0