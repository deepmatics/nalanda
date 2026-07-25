from pathlib import Path
from typing import Any, Dict
import yaml

class YamlFile:
    @staticmethod
    def load(file_path: str | Path) -> Dict[str, Any]:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)