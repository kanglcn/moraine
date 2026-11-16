"""internal utilities for CLI developing"""


__all__ = ['mk_clean_dir']

import datetime
import re
from pathlib import Path
import shutil

import numpy as np

def mk_clean_dir(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True,exist_ok=True)

def write_toml(path, table):
    """Write a TOML file of top-level keys and their values.

    Parameters
    ----------
    path : str or Path
        output: the TOML file (UTF-8)
    table : dict
        key (str) -> value: a string, boolean, integer, float, date or date-time (with its UTC offset, if any), or a
        list, tuple or numpy array of them; numpy scalars are written as the Python numbers they convert to (a
        float32 as the float64 of the same value)

    Returns
    -------
    str
        the text written
    """
    text = ''.join(f'{_toml_key(key)} = {_toml_value(value)}\n' for key, value in table.items())
    Path(path).write_text(text, encoding='utf-8')
    return text

def _toml_key(key):
    return key if re.fullmatch(r'[A-Za-z0-9_-]+', key) else _toml_str(key)

def _toml_str(s):
    """basic string: backslash, quote and the control characters escaped"""
    s = s.replace('\\', '\\\\').replace('"', '\\"')
    return '"' + re.sub(r'[\x00-\x1f\x7f]', lambda m: f'\\u{ord(m.group()):04x}', s) + '"'

def _toml_value(v):
    if isinstance(v, (bool, np.bool_)):
        return 'true' if v else 'false'
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return repr(float(v))       # the shortest text of the float64 value; nan, inf and -inf are TOML floats too
    if isinstance(v, str):
        return _toml_str(v)
    if isinstance(v, datetime.date):   # also datetime.datetime; RFC 3339, the UTC offset +00:00 as Z
        return v.isoformat().replace('+00:00', 'Z')
    if isinstance(v, (list, tuple, np.ndarray)):
        return '[' + ', '.join(_toml_value(u) for u in v) + ']'
    raise TypeError(f'write_toml: no TOML value for {type(v).__name__} {v!r}')
