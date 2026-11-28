"""The agglpy logger.

`logger` (``"agglpy"``) has only a ``NullHandler``, so agglpy writes no
log output until the application using it configures logging.
"""

import logging

logger: logging.Logger = logging.getLogger("agglpy")
"""The package logger, silent until logging is configured."""
logger.addHandler(logging.NullHandler())
