"""Package resources and the explicitly source-bound execution checkout.

Scientific model imports work from an installed wheel. Source shipping, tests and
experiment execution additionally require a checkout, never a sibling worktree.
"""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
RESOURCES = PACKAGE_ROOT / 'resources'
ROOT = PACKAGE_ROOT.parents[1]


def require_checkout():
    if not (ROOT / 'pyproject.toml').is_file() or not (ROOT / 'src/hope_cad').samefile(PACKAGE_ROOT):
        raise RuntimeError('This command needs the verified source checkout. Run pixi install --locked; model APIs also work from a wheel.')
    return ROOT
