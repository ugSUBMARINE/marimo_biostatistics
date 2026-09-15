"""Load course datasets in native Python and browser-based notebooks."""

import sys
from io import BytesIO
from urllib.request import urlopen

import pandas as pd


def read_csv(path, **kwargs):
    """Read a CSV without decompressing browser HTTP responses twice.

    In Pyodide, the browser decodes HTTP gzip content, but the response can
    retain its Content-Encoding header. Passing the URL directly to pandas
    makes pandas apply gzip decompression again. A byte buffer bypasses its
    HTTP-header handling while preserving CSV options such as encoding and
    comment. Native Python keeps pandas' normal file/URL handling.
    """
    if sys.platform == "emscripten" and str(path).startswith(("https://", "http://")):
        with urlopen(str(path)) as response:
            return pd.read_csv(BytesIO(response.read()), **kwargs)
    return pd.read_csv(path, **kwargs)
