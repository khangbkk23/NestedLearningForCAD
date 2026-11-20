"""Windows shim: Khang's nar_pilot_00/full.py imports the Unix-only `resource` module at top level (RAM accounting).
BASE generation never calls it; install an inert stub that raises if anything does, then run the B326 driver unchanged."""
import runpy
import sys
import types

stub = types.ModuleType('resource')


def _unavailable(*a, **k):
    raise RuntimeError('resource is unavailable on Windows (stub)')


stub.getrusage = _unavailable
stub.RUSAGE_SELF = 0
sys.modules['resource'] = stub
driver = sys.argv[1]
sys.path.insert(0, str(__import__('pathlib').Path(driver).parent))
sys.argv = [driver, *sys.argv[2:]]
runpy.run_path(driver, run_name='__main__')
