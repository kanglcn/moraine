"""Basic math routine"""


__all__ = ['math']

import logging
import zarr

from ..api.utils_.chunk import all_chunk_slices
from ..api.utils_.expr import evaluate
from .logging import mc_logger
from .executor import Executor, Chunk


def _math(*arrays, names, operation):
    return evaluate(operation, dict(zip(names, arrays)))

@mc_logger
def math(output:str,
         operation:str,
         **data):
    """Basic math manipulation. Only elementwise operations are supported. Only one output is supported.

    Parameters
    ----------
    output : str
        output: path of the result, same shape and chunks as the inputs, the dtype of `operation`
    operation : str
        elementwise expression of the input arrays, e.g. 'sin(a)*exp(b)/2' or '(a>=0)&(a<=0.3)': numbers, the
        names of the arrays, parentheses, + - * / ** % << >>, the comparisons < <= > >= == !=, & | ~ (and, or, not
        of bool arrays) and the functions where(condition, x, y), abs, sqrt, exp, expm1, log, log10, log1p, sin,
        cos, tan, arcsin, arccos, arctan, arctan2, sinh, cosh, tanh, arcsinh, arccosh, arctanh, floor, ceil, fmod,
        real, imag, conj, complex, isnan, isinf, isfinite, copy, ones_like. The dtype follows numpy's rules: a
        number does not change the dtype of an array (float32 times 2.5 is float32), a comparison is bool
    **data
        input: the arrays used in `operation` as NAME=zarr path, e.g. a='a.zarr' (on the command line:
        --kw a=a.zarr)
    """
    logger = logging.getLogger(__name__)
    names = list(data)
    zarrs = []
    for name, path in data.items():
        z = zarr.open(path, mode='r'); logger.zarr_info(name, z); zarrs.append(z)
    z0 = zarrs[0]
    for name, z in zip(names, zarrs):
        if z.shape != z0.shape:
            raise ValueError(f'{name}: shape {z.shape} differs from {names[0]}: {z0.shape}')
    # the dtype of the result from the expression on one element of every input
    dtype = evaluate(operation, {n: z[(0,) * z.ndim][None] for n, z in zip(names, zarrs)}).dtype
    out_zarr = zarr.open(output, mode='w', shape=z0.shape, dtype=dtype, chunks=z0.chunks)
    logger.zarr_info(output, out_zarr)
    tasks = [([Chunk(path, sl) for path in data.values()], [Chunk(output, sl)]) for sl in all_chunk_slices(z0.shape, z0.chunks)]
    logger.info(f'{operation} on {len(tasks)} chunks')
    with Executor(threads_per_worker=2) as ex:
        ex.map_chunks(_math, tasks, desc='math', names=names, operation=operation)
    logger.info('done.')
