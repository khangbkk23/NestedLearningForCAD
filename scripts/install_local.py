"""Install this package into the selected existing local environment."""
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
try:
    import pip
except ImportError:
    subprocess.run([sys.executable, '-m', 'ensurepip'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', str(root),
                '--no-deps', '--no-build-isolation'], check=True)
