from __future__ import annotations

import importlib.util
import inspect
import sys
import tempfile
from pathlib import Path


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _call_test(function) -> None:
    kwargs = {}
    signature = inspect.signature(function)
    for name in signature.parameters:
        if name == "tmp_path":
            temp_dir = tempfile.TemporaryDirectory()
            kwargs[name] = Path(temp_dir.name)
            try:
                function(**kwargs)
            finally:
                temp_dir.cleanup()
            return
        raise RuntimeError(f"Unsupported fixture: {name}")
    function(**kwargs)


def main() -> int:
    repo_root = Path.cwd()
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    tests_dir = repo_root / "tests"
    files = sorted(tests_dir.glob("test_*.py"))
    failures = []

    for file_path in files:
        module = _load_module(file_path)
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("test_"):
                try:
                    _call_test(obj)
                    print(f"PASS {file_path.name}::{name}")
                except Exception as exc:  # pragma: no cover
                    failures.append((file_path.name, name, exc))
                    print(f"FAIL {file_path.name}::{name}: {exc}")

    if failures:
        return 1
    print(f"{len(files)} files checked successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
