"""Own-checkout tests with deterministic math and permitted Windows scratch."""
import os,platform,sys,tempfile
from .runtime import ROOT,configure_runtime

def main():
    folder=ROOT/'.pytest_cache/portable-tests';folder.mkdir(parents=True,exist_ok=True)
    for name in ('TEMP','TMP','TMPDIR'):os.environ[name]=str(folder)
    tempfile.tempdir=str(folder);platform.platform();configure_runtime()
    import pytest
    return pytest.main(['-q',*(sys.argv[1:] or ['tests'])])

if __name__=='__main__':raise SystemExit(main())
