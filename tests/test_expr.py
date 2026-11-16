import re
import warnings

import numpy as np
import pytest

from moraine.api.utils_.expr import evaluate
from moraine.api.utils_ import ArgumentError

_rng = np.random.default_rng(0)
F = _rng.normal(size=(30, 40)).astype(np.float32)
G = _rng.normal(size=(30, 40)).astype(np.float32)
I = _rng.integers(-50, 50, size=(30, 40)).astype(np.int32)   # noqa: E741
J = _rng.integers(1, 9, size=(30, 40)).astype(np.int32)
C = (F + 1j * G).astype(np.complex64)
ARRAYS = {'f': F, 'g': G, 'i': I, 'j': J, 'c': C}


# the reference is numpy on the arrays with Python numbers: the same values and dtypes
@pytest.mark.parametrize('expression, reference', [
    ('f < 0.4', lambda: F < 0.4),
    ('(f >= -0.5) & (f <= 0.5)', lambda: (F >= -0.5) & (F <= 0.5)),
    ('(f < -1) | ~(g > 1)', lambda: (F < -1) | ~(G > 1)),
    ('i >= 8', lambda: I >= 8),
    ('i != j', lambda: I != J),
    ('i * 3 - j', lambda: I * 3 - J),
    ('i % j', lambda: np.remainder(I, J)),
    ('i << 2', lambda: I << 2),
    ('i >> 1', lambda: I >> 1),
    ('i / j', lambda: I / J),
    ('-f + +g', lambda: -F + G),
    ('f ** 2', lambda: F ** 2),
    ('f * 2.5', lambda: F * 2.5),
    ('f * -2.5', lambda: F * -2.5),                # a negative number stays a Python number: float32
    ('f * sqrt(2)', lambda: F * 2 ** 0.5),          # so does a function of numbers
    ('sin(f) * exp(g) / 2', lambda: np.sin(F) * np.exp(G) / 2),
    ('where(f > 0, f, 0)', lambda: np.where(F > 0, F, 0)),
    ('arctan2(g, f)', lambda: np.arctan2(G, F)),
    ('abs(c)', lambda: np.abs(C)),
    ('real(c * conj(c))', lambda: np.real(C * np.conj(C))),
    ('imag(c)', lambda: np.imag(C)),
    ('complex(f, g)', lambda: C),
    ('isnan(log(f))', lambda: np.isnan(np.log(F))),
    ('fmod(i, j)', lambda: np.fmod(I, J)),
    ('floor(f) + ceil(g)', lambda: np.floor(F) + np.ceil(G)),
    ('1 + 2 * 3', lambda: np.asarray(7)),
])
def test_values_and_dtypes(expression, reference):
    with np.errstate(all='ignore'):
        expected = np.asarray(reference())
    out = evaluate(expression, ARRAYS)
    assert (out.dtype, out.shape) == (expected.dtype, expected.shape)
    np.testing.assert_array_equal(out, expected)


def test_float32_stays_float32():
    for expression in ['f * 2.5', '-2.5 * f + 1', 'f / 3', 'where(f > 0, f, -1.5)', 'abs(c)', 'real(c) * 0.5']:
        assert evaluate(expression, ARRAYS).dtype == np.float32, expression


def test_complex_with_infinite_parts():
    out = evaluate('complex(a, b)', {'a': np.array([1.0, np.inf]), 'b': np.array([np.inf, 2.0])})
    np.testing.assert_array_equal(out.real, [1.0, np.inf])
    np.testing.assert_array_equal(out.imag, [np.inf, 2.0])


def test_no_warnings_for_invalid_values():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        out = evaluate('log(f) / (g - g)', ARRAYS)
    assert np.isnan(out).any() and np.isinf(out).any()


@pytest.mark.parametrize('expression, message', [
    ('f >', 'invalid syntax'),
    ('x > 1', "unknown name 'x'"),
    ('(f > 0) and (g > 0)', 'use & / | / ~'),
    ('not f', 'use & / | / ~'),
    ('0 < f < 1', 'chained comparison'),
    ('f is g', 'the comparison Is'),
    ('sum(f)', 'sum()'),
    ('max(f)', 'max()'),
    ('mean(f)', 'the function mean()'),
    ('f.sum()', 'the function f.sum()'),
    ('f.__class__', 'Attribute'),
    ('__import__("os").system("true")', 'the function'),
    ('f[0]', 'Subscript'),
    ('f // 2', 'FloorDiv'),
    ('i ^ j', 'BitXor'),
    ('"abc" == f', "the constant 'abc'"),
    ('lambda: 1', 'Lambda'),
    ('where(f > 0, x=1, y=2)', 'keyword arguments of where()'),
])
def test_errors(expression, message):
    with pytest.raises(ArgumentError, match=re.escape(message)):
        evaluate(expression, ARRAYS)
