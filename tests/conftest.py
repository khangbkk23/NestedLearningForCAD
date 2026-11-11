"""Tests use disposable local cache; no data downloads or GPU initialization."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from nested_replay_cad import paths
paths.process_environment()
