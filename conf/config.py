import yaml
import os
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "reference" / "phase1_baseline.yaml"


def load_config(config_path=None):
    config_path = str(config_path or DEFAULT_CONFIG_PATH)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Cannot find config file at {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    print(f"Config loading successfully {config_path}")
    return config
