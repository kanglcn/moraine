"""Elementwise expressions of named arrays, e.g. ``(ras >= 0) & (ras <= 0.3)``: the `operation` of the commands
`math`, `pc-logic-ras` and `pc-logic-pc` (decision 0045)."""

__all__ = ['evaluate']

import ast

import numpy as np

_BINARY = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.true_divide, ast.Pow: np.power,
           ast.Mod: np.remainder, ast.LShift: np.left_shift, ast.RShift: np.right_shift, ast.BitAnd: np.bitwise_and,
           ast.BitOr: np.bitwise_or}
_UNARY = {ast.USub: np.negative, ast.UAdd: np.positive, ast.Invert: np.invert}
_COMPARE = {ast.Lt: np.less, ast.LtE: np.less_equal, ast.Gt: np.greater, ast.GtE: np.greater_equal, ast.Eq: np.equal,
            ast.NotEq: np.not_equal}


def _complex(re, im):
    re, im = np.asarray(re), np.asarray(im)
    out = np.empty(np.broadcast_shapes(re.shape, im.shape), np.result_type(re, im, np.complex64))
    out.real, out.imag = re, im     # not re + 1j * im: an infinite imaginary part would make the real part nan
    return out


_FUNCTIONS = {name: getattr(np, name) for name in (
    'where', 'abs', 'sqrt', 'exp', 'expm1', 'log', 'log10', 'log1p', 'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
    'arctan2', 'sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh', 'floor', 'ceil', 'fmod', 'real', 'imag', 'conj',
    'isnan', 'isinf', 'isfinite', 'copy', 'ones_like')}
_FUNCTIONS['complex'] = _complex

_NUMBER = (bool, int, float, complex)   # Python numbers, exact types (numpy scalars subclass float and complex)


def _apply(fn, *args):
    out = fn(*args)
    if all(type(a) in _NUMBER for a in args):
        # numbers only: the result stays a Python number, which does not change the dtype of an array it meets
        # later (numpy scalars do: float32 array x float64 scalar is float64)
        out = np.asarray(out)[()].item()
    return out


def _unsupported(node):
    if isinstance(node, ast.BoolOp) or isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return 'and / or / not: use & / | / ~ with parentheses, e.g. (a > 0) & (a < 1)'
    if isinstance(node, ast.Compare) and len(node.ops) > 1:
        return 'a chained comparison: use (a < b) & (b < c)'
    if isinstance(node, ast.Compare):
        return f'the comparison {type(node.ops[0]).__name__}'
    if isinstance(node, ast.Call):
        name = node.func.id if isinstance(node.func, ast.Name) else None
        if name in ('sum', 'prod', 'min', 'max'):
            return f'{name}(): the value of an element depends on that element only'
        if name in _FUNCTIONS:
            return f'keyword arguments of {name}()'
        return f'the function {name or ast.unparse(node.func)}()'
    if isinstance(node, ast.BinOp):
        return f'the operator {type(node.op).__name__}'
    if isinstance(node, ast.Constant):
        return f'the constant {node.value!r}'
    return f'{type(node).__name__} ({ast.unparse(node)})'


def _eval(node, arrays, expression):
    def ev(n):
        return _eval(n, arrays, expression)
    if isinstance(node, ast.Constant) and type(node.value) in _NUMBER:
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in arrays:
            raise ValueError(f'expression {expression!r}: unknown name {node.id!r}; the arrays are '
                             f'{", ".join(arrays) or "none"}')
        return arrays[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return _apply(_BINARY[type(node.op)], ev(node.left), ev(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _apply(_UNARY[type(node.op)], ev(node.operand))
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _COMPARE:
        return _apply(_COMPARE[type(node.ops[0])], ev(node.left), ev(node.comparators[0]))
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS
            and not node.keywords):
        return _apply(_FUNCTIONS[node.func.id], *[ev(a) for a in node.args])
    raise ValueError(f'expression {expression!r}: not supported: {_unsupported(node)}')


def evaluate(expression:str, arrays:dict)->np.ndarray:
    """Evaluate an elementwise expression of named arrays.

    Parameters
    ----------
    expression : str
        the expression, e.g. ``'(ras >= 0) & (ras <= 0.3)'`` or ``'sin(a) * exp(b) / 2'``: numbers, the names of
        `arrays`, parentheses, the operators ``+ - * / ** %`` (``%``: the remainder with the sign of the divisor),
        ``<< >>``, the comparisons ``< <= > >= == !=``, ``& | ~`` (and, or, not of bool arrays; bitwise of integer
        arrays) and the functions ``where(condition, x, y)``, ``abs``, ``sqrt``, ``exp``, ``expm1``, ``log``, ``log10``,
        ``log1p``, ``sin``, ``cos``, ``tan``, ``arcsin``, ``arccos``, ``arctan``, ``arctan2(y, x)``, ``sinh``, ``cosh``,
        ``tanh``, ``arcsinh``, ``arccosh``, ``arctanh``, ``floor``, ``ceil``, ``fmod``, ``real``, ``imag``, ``conj``,
        ``complex(real, imag)``, ``isnan``, ``isinf``, ``isfinite``, ``copy``, ``ones_like``
    arrays : dict
        name -> numpy array of every name used in `expression`; the arrays have the same shape or shapes that
        broadcast

    Returns
    -------
    np.ndarray
        the value of the expression at every element, of the broadcast shape of the arrays used. Its dtype follows
        numpy's rules: a number does not change the dtype of an array (float32 times 2.5 is float32), a comparison
        is bool. A division by zero or an invalid value gives inf or nan, without a warning
    """
    try:
        tree = ast.parse(expression, mode='eval')
    except SyntaxError as e:
        raise ValueError(f'expression {expression!r}: invalid syntax ({e.msg})') from None
    with np.errstate(all='ignore'):
        return np.asarray(_eval(tree.body, arrays, expression))
