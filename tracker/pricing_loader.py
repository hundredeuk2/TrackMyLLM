import yaml
from pathlib import Path
from typing import Dict

BASE_DIR = Path(__file__).parent
DEFAULT_PRICING_FILE = BASE_DIR / "pricing.yaml"

def load_pricing_yaml(path: str | Path | None = None) -> Dict[str, Dict[str, float]]:
    yaml_path = Path(path) if path else DEFAULT_PRICING_FILE
    if not yaml_path.is_absolute():
        yaml_path = BASE_DIR / yaml_path
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data
