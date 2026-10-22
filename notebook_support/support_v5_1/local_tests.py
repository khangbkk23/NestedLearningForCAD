"""Checkout test entrypoint with the same deterministic numerical setup."""
from pathlib import Path
import os
import platform
import sys
import tempfile
from .common import ROOT,configure_runtime


def main():
    folder=ROOT/'.pytest_cache'/'v5_1_tmp'
    folder.mkdir(parents=True,exist_ok=True)
    for name in ('TEMP','TMP','TMPDIR'):os.environ[name]=str(folder)
    tempfile.tempdir=str(folder)
    # A preserved test mocks subprocess process-wide; warm Python's cached
    # Windows OS probe before the mock rather than changing frozen test/source.
    platform.platform()
    configure_runtime()
    import pytest
    return pytest.main(['tests','-q',*sys.argv[1:]])


if __name__=='__main__':raise SystemExit(main())
