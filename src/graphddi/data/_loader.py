"""Helpers for chatty third-party data loaders."""

import contextlib
import io
import sys
from collections.abc import Iterator


@contextlib.contextmanager
def quiet_stderr() -> Iterator[None]:
    """Discard stderr writes inside the block; re-emit them if the block raises.

    Useful for libraries (TDC, …) that print progress chatter to stderr but
    also surface real errors through the same channel. We swallow the noise
    on success and preserve the diagnostics on failure.
    """
    buf = io.StringIO()
    try:
        with contextlib.redirect_stderr(buf):
            yield
    except BaseException:
        sys.stderr.write(buf.getvalue())
        raise
