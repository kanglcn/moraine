"""basic logging functions for the CLI"""


__all__ = ['McLogger', 'mc_logger', 'mc_logger_', 'get_logger']

import logging
import sys
import random
import string
import inspect
from functools import wraps
import types

import zarr

class McLogger(logging.getLoggerClass()):
    def zarr_info(self,
                  path,
                  zarr,
                 ):
        """Parameters
        ----------
        path
            string to zarr
        zarr
            zarr dataset
        """
        self.info(f'{path} zarray shape, chunks, dtype: {zarr.shape}, {zarr.chunks}, {zarr.dtype}')

def mc_logger(func):
    @wraps(func)
    def log_args(*args, **kwargs):
        logging.setLoggerClass(McLogger)
        logger = logging.getLogger(__name__)
        logger.info(f'running function: {func.__name__}')
        ba = inspect.signature(func).bind(*args, **kwargs)
        ba.apply_defaults()
        func_args = ba.arguments
        func_args_strs = map("{0[0]} = {0[1]!r}".format, func_args.items())
        logger.info('fetching args:')
        for item in func_args_strs:
            logger.info(item)
        logger.info('fetching args done.')
        return func(*args, **kwargs)
    return log_args

def mc_logger_(func):
    @wraps(func)
    def log_args(*args, **kwargs):
        logging.setLoggerClass(McLogger)
        return func(*args, **kwargs)
    return log_args

def get_logger(logfile:str=None,
              ):
    """get logger for decorrelation cli application

    Parameters
    ----------
    logfile : str, optional
        logfile, optional. default: no logfile
    """

    level = logging.INFO
    logger = logging.getLogger()
    # print(logger.zarr_info)
    logger.setLevel(level)
    formatter = logging.Formatter(f'%(asctime)s - %(funcName)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if logfile:
        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
