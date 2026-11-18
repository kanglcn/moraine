"""Times of images: date-times in UTC from the forms users give them in"""

import datetime

import numpy as np

from .errors import ArgumentError


def _utc(t):
    """a time as a naive datetime in UTC, from an aware datetime, a naive one (taken as UTC), ISO 8601 text or a
    numpy.datetime64"""
    raw = t
    if isinstance(t, np.datetime64):
        t = t.astype('datetime64[us]').astype(datetime.datetime)
    elif isinstance(t, str):
        try:                # with the time of the day: a date alone (e.g. YYYYMMDD) is not the time of an image
            t = datetime.datetime.fromisoformat(t.strip().replace('Z', '+00:00').replace('z', '+00:00')) \
                if ':' in t else None
        except ValueError:
            t = None
    if not isinstance(t, datetime.datetime):
        raise ArgumentError(f'times must be date-times in UTC (ISO 8601, e.g. 2019-01-02T05:06:07.123456Z), '
                            f'got {raw!r}')
    if t.tzinfo is not None:
        t = t.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return t


def _times(times)->list:
    """naive UTC datetimes of a list of times, of a single time, or of text with commas or spaces"""
    if isinstance(times, str):
        times = times.replace(',', ' ').split()
    elif isinstance(times, (datetime.datetime, np.datetime64)):
        times = [times]
    return [_utc(t) for t in times]
