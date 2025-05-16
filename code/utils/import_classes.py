import importlib

import yaml


def load_class(name: str, config_path: str = "/home/thilsk/Documents/12_Projects/12.02_BIL_project/6_BIL_FL/code/config/class_config.yaml"):
    with open(config_path, "r", encoding = 'utf-8') as f:
        config = yaml.safe_load(f)
    
    class_info = config.get(name)
    if not class_info:
        raise ValueError(f"No config found for class '{name}'")
    
    module = importlib.import_module(class_info["module"])
    cls = getattr(module, class_info["class"])
    return cls
