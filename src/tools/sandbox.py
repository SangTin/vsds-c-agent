from __future__ import annotations

import subprocess
import sys
import tempfile


_PREAMBLE = r"""
import builtins
import decimal
import fractions
import itertools
import math
import statistics

try:
    import sympy
    import sympy as sp
    from sympy import *
except Exception:
    sympy = None
    sp = None

_DENIED_IMPORTS = {"os", "sys", "subprocess"}

def _compute_only_import(name, globals=None, locals=None, fromlist=(), level=0, _real_import=builtins.__import__):
    root = name.split(".", 1)[0]
    if root in _DENIED_IMPORTS:
        raise ImportError(f"Import of {root!r} is disabled in the compute sandbox")
    return _real_import(name, globals, locals, fromlist, level)

def _blocked_open(*args, **kwargs):
    raise RuntimeError("open() is disabled in the compute sandbox")

# The generated snippets are for computation only; file/process access is blocked
# after the numeric libraries are loaded, and the code still runs in -I mode.
builtins.__import__ = _compute_only_import
builtins.open = _blocked_open
"""


def run_python(code: str, timeout: float = 5.0) -> dict[str, bool | str]:
    """Run a compute-only Python snippet in an isolated subprocess."""
    program = f"{_PREAMBLE}\n{code}"
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", program],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr
            message = f"Timeout after {timeout} seconds"
            if stderr:
                message = f"{stderr.strip()}\n{message}"
            return {
                "ok": False,
                "stdout": (stdout or "").strip(),
                "stderr": message.strip(),
            }

    return {
        "ok": completed.returncode == 0,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
