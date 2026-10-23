"""Compatibility entrypoint: installation is managed only by Pixi."""
from pathlib import Path
import subprocess
if __name__ == '__main__':
    subprocess.run(['pixi','install','--locked'],cwd=Path(__file__).resolve().parents[1],check=True)
