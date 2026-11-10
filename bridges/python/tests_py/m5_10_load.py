"""Loads #108 part 2's modules by path, so another study's `run` or `report` cannot shadow them."""

import importlib.util
import pathlib
import sys

HERE = pathlib.Path(__file__).parents[1] / "experiments" / "m5_10"
NAMES = ("grid", "stats", "run", "report")
_loaded = {}


def module(name):
    if name in _loaded:
        return _loaded[name]
    saved = {n: sys.modules.pop(n) for n in NAMES if n in sys.modules}
    sys.path.insert(0, str(HERE))
    try:
        for n in NAMES:
            spec = importlib.util.spec_from_file_location(n, HERE / f"{n}.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[n] = mod
            spec.loader.exec_module(mod)
            _loaded[n] = mod
    finally:
        sys.path.remove(str(HERE))
        for n in NAMES:
            sys.modules.pop(n, None)
        sys.modules.update(saved)
    return _loaded[name]
