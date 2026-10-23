"""Package resources and the explicitly source-bound execution checkout.

Scientific model imports work from an installed wheel. Source shipping, tests and
experiment execution additionally require a checkout, never a sibling worktree.
"""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
RESOURCES = PACKAGE_ROOT / 'resources'
ROOT = PACKAGE_ROOT.parents[1]
WORKSPACE = ROOT / 'workspace'
DATA = WORKSPACE / 'data'
WEIGHTS = WORKSPACE / 'weights'
RECORDS = WORKSPACE / 'records'
CHECKS = WORKSPACE / 'checks'
EXPERIMENTS = WORKSPACE / 'experiments'


def session_export_directory(work):
    """Reserve a fresh immutable export slot; caller holds the run writer lock."""
    work = Path(work).resolve()
    base = work.parent / 'exports' if work.name == 'run' else work.parent / (work.name + '_exports')
    base.mkdir(parents=True, exist_ok=True)
    if base.is_symlink():
        raise ValueError('Export directory cannot be a symlink')
    for number in range(1, 1000000):
        slot = base / f'session_{number:02d}'
        try:
            slot.mkdir()
            return slot
        except FileExistsError:
            continue
    raise ValueError('No unused export session slot')


def require_checkout():
    if not (ROOT / 'pyproject.toml').is_file() or not (ROOT / 'src/hope_cad').samefile(PACKAGE_ROOT):
        raise RuntimeError('This command needs the verified source checkout. Run pixi install --locked; model APIs also work from a wheel.')
    return ROOT
