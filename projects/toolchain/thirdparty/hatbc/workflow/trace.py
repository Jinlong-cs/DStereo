import functools
import inspect
import types
from types import MethodType


def execute_and_trace():
    pass


def get_current_graph_tracer():
    pass


def _make_function_traceable(fn, *, allow_input_skip=False, trace_fn=None):
    """Wrap function `fn` to make it traceable under GraphTracer scope
    (no-op when outside it).
    """

    @functools.wraps(fn)
    def _wrap(*args, **kwargs):
        """This function is wrapped by `hatbc.workflow` to make it traceable
        under GraphTracer scope (no-op when outside it).
        """
        graph_tracer = get_current_graph_tracer()
        if graph_tracer is None:
            return fn(*args, **kwargs)
        else:
            return trace_fn(
                fn,
                args,
                kwargs,
                __workflow_allow_input_skip__=allow_input_skip,
            )  # noqa

    _wrap.__workflow_traceable__ = True
    _wrap.__workflow_allow_input_skip__ = allow_input_skip
    return _wrap


def _make_class_traceable(
    obj, *, return_patch_func_only=False, allow_input_skip=False, trace_fn=None
):  # noqa
    """Wrap __call__ function of class `obj` to make it traceable under
    GraphTracer scope (no-op when outside it).
    """

    assert isinstance(
        obj.__call__, types.FunctionType
    ), "the __call__ function of object should not be static"

    fn = obj.__call__

    @functools.wraps(fn)
    def _wrap(self, *args, **kwargs):
        """This function is wrapped by `hatbc.workflow` to make it traceable
        under GraphTracer scope (no-op when outside it).
        """
        graph_tracer = get_current_graph_tracer()
        if graph_tracer is None:
            return fn(self, *args, **kwargs)
        else:
            return trace_fn(
                self,
                args,
                kwargs,
                __workflow_allow_input_skip__=allow_input_skip,
            )  # noqa

    _wrap.__workflow_traceable__ = True
    _wrap.__workflow_allow_input_skip__ = allow_input_skip

    if return_patch_func_only:
        return _wrap
    else:
        obj.__call__ = _wrap
        return obj


def make_traceable(obj=None, *, allow_input_skip=False, trace_fn=None):
    """A decorator that make a callable object to be an workflow operator.

    Parameters
    ----------
    obj : callable, optional
        Wrapped callable object, by default None
    allow_input_skip : bool, optional
        Whether allow input is skip, for more information, please see the workflow tutorial, by default False
    """  # noqa

    def _impl(obj):
        assert callable(obj), f"{obj} should be callable"

        if inspect.isfunction(obj) or isinstance(obj, MethodType):
            return _make_function_traceable(
                obj, allow_input_skip=allow_input_skip, trace_fn=trace_fn
            )
        else:
            return _make_class_traceable(
                obj, allow_input_skip=allow_input_skip, trace_fn=trace_fn
            )

    if obj is None:
        return _impl
    else:
        return _impl(obj)
