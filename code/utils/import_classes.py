import importlib
from pathlib import Path

import yaml
from config.config import BASE_PATH

# Define config path 
config_dir = Path(BASE_PATH) / "code" / "config"
user_config = config_dir / "class_config.yaml"
default_config = config_dir / "class_config.yaml.example"

# Try loading user config first, then fallback
if user_config.exists():
    CONFIG_PATH = user_config
elif default_config.exists():
    print("Warning: 'class_config.yaml' not found. Using fallback 'class_config.yaml.example'")
    CONFIG_PATH = default_config
else:
    raise FileNotFoundError("No configuration file found. Expected 'class_config.yaml' or fallback 'class_config.yaml.example' in 'code/config/'.")


def load_class(name: str, config_path: Path = CONFIG_PATH):
    with open(config_path, "r", encoding = 'utf-8') as f:
        config = yaml.safe_load(f)
    
    class_info = config.get(name)
    if not class_info:
        raise ValueError(f"No config found for class '{name}'")
    
    module = importlib.import_module(class_info["module"])
    cls = getattr(module, class_info["class"])
    return cls
