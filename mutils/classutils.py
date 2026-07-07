"""
The MIT License (MIT)

Copyright (c) 2026 Marvin Teichmann

Lightweight, general-purpose helpers for working with classes and instances.

This module is intended as a small home for class-level conveniences that do
not warrant their own module. It currently provides:

* :func:`summarize_value` -- a compact one-line ``type (+ shape / len / value)``
  description of an arbitrary value (handy for logging/debugging).
* :class:`SelfDescribing` -- a mixin giving plain (data)classes a friendly
  interactive summary via ``obj.keys()`` / ``obj.help()``.

Everything here is dependency-light: ``numpy`` is used only to prettify array
summaries and is optional (the module degrades gracefully without it). Add
further small class utilities here as the need arises.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dataclasses import fields, is_dataclass

try:  # numpy is optional -- only used to prettify ndarray/scalar summaries
    import numpy as np
except ImportError:  # pragma: no cover - numpy is normally available
    np = None


__all__ = ["summarize_value", "SelfDescribing"]


# Scalar types rendered inline as ``type = value``. numpy scalar bases are
# appended when numpy is available.
_SCALAR_TYPES: tuple = (bool, int, float)
if np is not None:
    _SCALAR_TYPES = _SCALAR_TYPES + (np.integer, np.floating)


def summarize_value(value, maxlen: int = 56) -> str:
    """Return a short ``type (+ shape / len / value)`` description of *value*.

    Renders a compact, single-line summary suitable for interactive listings
    and logging: arrays show their shape/dtype, containers their length (dicts
    also preview a few keys), scalars and short strings their value, and
    anything else just its type name.
    """
    if value is None:
        return "None"
    if np is not None and isinstance(value, np.ndarray):
        return f"ndarray {tuple(value.shape)} {value.dtype}"
    if isinstance(value, dict):
        preview = ", ".join(str(k) for k in list(value)[:6])
        if len(value) > 6:
            preview += ", ..."
        body = f" {{{preview}}}" if value else ""
        return f"dict[{len(value)}]{body}"
    if isinstance(value, (list, tuple, set)):
        return f"{type(value).__name__}[{len(value)}]"
    if isinstance(value, _SCALAR_TYPES):
        return f"{type(value).__name__} = {value}"
    if isinstance(value, (str, bytes)):
        text = repr(value)
        if len(text) > maxlen:
            text = text[: maxlen - 4] + "..." + text[-1]
        return f"{type(value).__name__} = {text}"
    return type(value).__name__


class SelfDescribing:
    """Mixin adding an interactive member/type summary.

    ``obj.keys()`` (aliased as ``obj.help()``) prints each data member with a
    short type/shape summary and lists the available properties/methods (names
    only, never evaluated), so the object is easy to navigate in IPython/ipdb.
    It returns the list of data-member names.

    Works with both :mod:`dataclasses` (fields are used as the data members)
    and plain classes (public instance attributes are used).
    """

    def keys(self) -> list:
        cls = type(self)

        if is_dataclass(self):
            names = [f.name for f in fields(self)]
        else:
            names = [n for n in vars(self) if not n.startswith("_")]

        width = max((len(n) for n in names), default=0)
        lines = [f"{cls.__name__}:"]
        for name in names:
            summary = summarize_value(getattr(self, name, None))
            lines.append(f"  {name:<{width}} : {summary}")

        # Properties/methods listed by name only (not evaluated), so expensive
        # or argument-taking members stay discoverable without side effects.
        api = []
        for name in dir(cls):
            if (
                name.startswith("_")
                or name in ("keys", "help")
                or name in names
            ):
                continue
            attr = getattr(cls, name, None)
            if isinstance(attr, property):
                api.append(name)
            elif callable(attr):
                api.append(f"{name}()")
        if api:
            lines.append("  api: " + ", ".join(sorted(api)))

        print("\n".join(lines))
        return names

    # ``help`` is a friendlier alias for interactive use.
    help = keys
