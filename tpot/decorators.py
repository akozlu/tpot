# -*- coding: utf-8 -*-

"""This file is part of the TPOT library.

TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - and many more generous open source contributors

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.

"""

from __future__ import print_function
from functools import wraps
import warnings
from .export_utils import expr_to_tree, generate_pipeline_code
from deap import creator
import signal
import ctypes
import threading
from logging import NullHandler
import functools
import logging
import sys
# Custom logger
LOG = logging.getLogger(name='stopit')
tid_ctype = ctypes.c_ulong
LOG.addHandler(NullHandler())

NUM_TESTS = 10
MAX_EVAL_SECS = 10

class TimeoutException(Exception):
    """Raised when the block under context management takes longer to complete
    than the allowed maximum timeout value.
    """
    pass

class BaseTimeout(object):
    """Context manager for limiting in the time the execution of a block
    :param seconds: ``float`` or ``int`` duration enabled to run the context
      manager block
    :param swallow_exc: ``False`` if you want to manage the
      ``TimeoutException`` (or any other) in an outer ``try ... except``
      structure. ``True`` (default) if you just want to check the execution of
      the block with the ``state`` attribute of the context manager.
    """
    # Possible values for the ``state`` attribute, self explanative
    EXECUTED, EXECUTING, TIMED_OUT, INTERRUPTED, CANCELED = range(5)

    def __init__(self, seconds, swallow_exc=True):
        self.seconds = seconds
        self.swallow_exc = swallow_exc
        self.state = BaseTimeout.EXECUTED

    def __bool__(self):
        return self.state in (BaseTimeout.EXECUTED, BaseTimeout.EXECUTING, BaseTimeout.CANCELED)

    __nonzero__ = __bool__  # Python 2.x

    def __repr__(self):
        """Debug helper
        """
        return "<{0} in state: {1}>".format(self.__class__.__name__, self.state)

    def __enter__(self):
        self.state = BaseTimeout.EXECUTING
        self.setup_interrupt()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is TimeoutException:
            if self.state != BaseTimeout.TIMED_OUT:
                self.state = BaseTimeout.INTERRUPTED
                self.suppress_interrupt()
            LOG.warning("Code block execution exceeded {0} seconds timeout".format(self.seconds),
                        exc_info=(exc_type, exc_val, exc_tb))
            return self.swallow_exc
        else:
            if exc_type is None:
                self.state = BaseTimeout.EXECUTED
            self.suppress_interrupt()
        return False

    def cancel(self):
        """In case in the block you realize you don't need anymore
       limitation"""
        self.state = BaseTimeout.CANCELED
        self.suppress_interrupt()

    # Methods must be provided by subclasses
    def suppress_interrupt(self):
        """Removes/neutralizes the feature that interrupts the executed block
        """
        raise NotImplementedError

    def setup_interrupt(self):
        """Installs/initializes the feature that interrupts the executed block
        """
        raise NotImplementedError


class base_timeoutable(object):  # noqa
    """A base for function or method decorator that raises a ``TimeoutException`` to
    decorated functions that should not last a certain amount of time.
    Any decorated callable may receive a ``timeout`` optional parameter that
    specifies the number of seconds allocated to the callable execution.
    The decorated functions that exceed that timeout return ``None`` or the
    value provided by the decorator.
    :param default: The default value in case we timed out during the decorated
      function execution. Default is None.
    :param timeout_param: As adding dynamically a ``timeout`` named parameter
      to the decorated callable may conflict with the callable signature, you
      may choose another name to provide that parameter. Your decoration line
      could look like ``@timeoutable(timeout_param='my_timeout')``
    .. note::
       This is a base class that must be subclassed. subclasses must override
       thz ``to_ctx_mgr`` with a timeout  context manager class which in turn
       must subclasses of above ``BaseTimeout`` class.
    """
    to_ctx_mgr = None

    def __init__(self, default=None, timeout_param='timeout'):
        self.default, self.timeout_param = default, timeout_param

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            timeout = kwargs.pop(self.timeout_param, None)
            if timeout:
                with self.to_ctx_mgr(timeout, swallow_exc=True):
                    result = self.default  # noqa
                    # ``result`` may not be assigned below in case of timeout
                    result = func(*args, **kwargs)
                return result
            else:
                return func(*args, **kwargs)
        return wrapper

def async_raise(target_tid, exception):
    """Raises an asynchronous exception in another thread.
    Read http://docs.python.org/c-api/init.html#PyThreadState_SetAsyncExc
    for further enlightenments.
    :param target_tid: target thread identifier
    :param exception: Exception class to be raised in that thread
    """
    # Ensuring and releasing GIL are useless since we're not in C
    # gil_state = ctypes.pythonapi.PyGILState_Ensure()
    ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid_ctype(target_tid),
                                                     ctypes.py_object(exception))
    # ctypes.pythonapi.PyGILState_Release(gil_state)
    if ret == 0:
        raise ValueError("Invalid thread ID {}".format(target_tid))
    elif ret > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid_ctype(target_tid), None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


class ThreadingTimeout(BaseTimeout):
    """Context manager for limiting in the time the execution of a block
    using asynchronous threads launching exception.
    See :class:`stopit.utils.BaseTimeout` for more information
    """
    def __init__(self, seconds, swallow_exc=True):
        super(ThreadingTimeout, self).__init__(seconds, swallow_exc)
        self.target_tid = threading.current_thread().ident
        self.timer = None  # PEP8

    def stop(self):
        """Called by timer thread at timeout. Raises a Timeout exception in the
        caller thread
        """
        self.state = BaseTimeout.TIMED_OUT
        async_raise(self.target_tid, TimeoutException)

    # Required overrides
    def setup_interrupt(self):
        """Setting up the resource that interrupts the block
        """
        self.timer = threading.Timer(self.seconds, self.stop)
        self.timer.start()

    def suppress_interrupt(self):
        """Removing the resource that interrupts the block
        """
        self.timer.cancel()


class threading_timeoutable(base_timeoutable):  #noqa
    """A function or method decorator that raises a ``TimeoutException`` to
    decorated functions that should not last a certain amount of time.
    this one uses ``ThreadingTimeout`` context manager.
    See :class:`.utils.base_timoutable`` class for further comments.
    """
    to_ctx_mgr = ThreadingTimeout


class SignalTimeout(BaseTimeout):
    """Context manager for limiting in the time the execution of a block
    using signal.SIGALRM Unix signal.
    See :class:`stopit.utils.BaseTimeout` for more information
    """
    def __init__(self, seconds, swallow_exc=True):
        seconds = int(seconds)  # alarm delay for signal MUST be int
        super(SignalTimeout, self).__init__(seconds, swallow_exc)

    def handle_timeout(self, signum, frame):
        self.state = BaseTimeout.TIMED_OUT
        raise TimeoutException('Block exceeded maximum timeout '
                               'value (%d seconds).' % self.seconds)

    # Required overrides
    def setup_interrupt(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def suppress_interrupt(self):
        signal.alarm(0)
        signal.signal(signal.SIGALRM, signal.SIG_DFL)


class signal_timeoutable(base_timeoutable):  #noqa
    """A function or method decorator that raises a ``TimeoutException`` to
    decorated functions that should not last a certain amount of time.
    this one uses ``SignalTimeout`` context manager.
    See :class:`.utils.base_timoutable`` class for further comments.
    """
    to_ctx_mgr = SignalTimeout


def _pre_test(func):
    """Check if the wrapped function works with a pretest data set.

    Reruns the wrapped function until it generates a good pipeline, for a max of
    NUM_TESTS times.

    Parameters
    ----------
    func: function
        The decorated function.

    Returns
    -------
    check_pipeline: function
        A wrapper function around the func parameter
    """
    @threading_timeoutable(default="timeout")
    def time_limited_call(func, *args):
        func(*args)

    @wraps(func)
    def check_pipeline(self, *args, **kwargs):
        bad_pipeline = True
        num_test = 0  # number of tests

        # a pool for workable pipeline
        while bad_pipeline and num_test < NUM_TESTS:
            # clone individual before each func call so it is not altered for
            # the possible next cycle loop
            args = [self._toolbox.clone(arg) if isinstance(arg, creator.Individual) else arg for arg in args]
            try:

                if func.__name__ == "_generate":
                    expr = []
                else:
                    expr = tuple(args)
                pass_gen = False
                num_test_expr = 0
                # to ensure a pipeline can be generated or mutated.
                while not pass_gen and num_test_expr < int(NUM_TESTS/2):
                    try:
                        expr = func(self, *args, **kwargs)
                        pass_gen = True
                    except:
                        num_test_expr += 1
                        pass
                # mutation operator returns tuple (ind,); crossover operator
                # returns tuple of (ind1, ind2)

                expr_tuple = expr if isinstance(expr, tuple) else (expr,)
                for expr_test in expr_tuple:
                    pipeline_code = generate_pipeline_code(
                        expr_to_tree(expr_test, self._pset),
                        self.operators
                    )
                    sklearn_pipeline = eval(pipeline_code, self.operators_context)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        time_limited_call(
                            sklearn_pipeline.fit,
                            self.pretest_X,
                            self.pretest_y,
                            timeout=MAX_EVAL_SECS,
                        )

                    bad_pipeline = False
            except BaseException as e:
                message = '_pre_test decorator: {fname}: num_test={n} {e}.'.format(
                    n=num_test,
                    fname=func.__name__,
                    e=e

                )
                # Use the pbar output stream if it's active
                self._update_pbar(pbar_num=0, pbar_msg=message)
            finally:
                num_test += 1

        return expr


    return check_pipeline
