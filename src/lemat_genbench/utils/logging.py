# Copyright 2025 Entalpic
import logging
import os
from pathlib import Path

from rich.logging import RichHandler


def get_cache_dir():
    """Get the cache directory path, creating it if it doesn't exist."""
    cache_dir = Path.home() / ".cache" / "lematerial_fetcher"
    if os.environ.get("LEMATERIALFETCHER_CACHE_DIR", None):
        cache_dir = Path(os.environ["LEMATERIALFETCHER_CACHE_DIR"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


class Logger:
    """Entaflownet's custom logger. Handles logging to the console.

    .. note::

        The logger is set up to use the ``RichHandler``, which provides colored output
        and rich tracebacks. See
        `here <https://rich.readthedocs.io/en/stable/logging.html>`_ for more information.

    Parameters
    ----------
    level : str, optional
        Logging level, by default "NOTSET"
    """

    def __init__(self, level: str = "NOTSET"):
        if os.environ.get("LEMATERIALFETCHER_LOG_DIR", None):
            log_file = os.path.join(
                os.environ.get("LEMATERIALFETCHER_LOG_DIR"), "lematerial_fetcher.log"
            )
            self.file_logger = logging.getLogger("lematerial_fetcher")
            self.file_logger.setLevel(level)
            self.file_logger.addHandler(logging.FileHandler(log_file))
        else:
            self.file_logger = None

        format = "%(message)s"
        datefmt = "[%X]"
        handlers = [RichHandler(rich_tracebacks=True)]
        self.term_logger = logging.getLogger("lematerial_fetcher")

        self.set_level(level)
        self.term_logger.handlers = handlers
        self.term_logger.propagate = False
        # Set format to include pathname and line number from the correct call site
        formatter = logging.Formatter(format, datefmt)
        for handler in self.term_logger.handlers:
            handler.setFormatter(formatter)

    def set_level(self, level: str):
        """Set the logging level to the terminal logger from string format.

        Parameters
        ----------
        level : str
            Logging level, by default "NOTSET", see
            `here <https://docs.python.org/3/library/logging.html#logging-levels>`_
            for more information.
        """
        assert level in logging._nameToLevel.keys(), f"Invalid logging level: {level}"
        self.level = level
        self.term_logger.setLevel(level)

    def info(self, message: str, *args, **kwargs):
        # Find caller's source location with stacklevel=2
        self.term_logger.info(message, stacklevel=2, *args, **kwargs)

    def debug(self, message: str, *args, **kwargs):
        self.term_logger.debug(message, stacklevel=2, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        self.term_logger.warning(message, stacklevel=2, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        self.term_logger.error(message, stacklevel=2, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs):
        self.term_logger.critical(message, stacklevel=2, *args, **kwargs)

    def fatal(self, message: str, *args, **kwargs):
        self.term_logger.fatal(message, stacklevel=2, *args, **kwargs)


logger = Logger(
    level="DEBUG" if os.environ.get("lemat_genbench_DEBUG", None) else "INFO"
)
