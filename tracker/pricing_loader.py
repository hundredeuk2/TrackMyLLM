import yaml
from pathlib import Path
from typing import Dict, Union

BASE_DIR = Path(__file__).parent
DEFAULT_PRICING_FILE = BASE_DIR / "pricing.yaml"

def load_pricing_yaml(path: Union[str, Path, None] = None) -> Dict[str, Dict[str, float]]:
    """
    Load pricing configuration from a YAML file.

    Args:
        path: Optional; path to a YAML file or directory. If None, falls back to DEFAULT_PRICING_FILE.

    Returns:
        A dict mapping provider keys to their prompt/completion pricing.
    """
    yaml_path = Path(path) if path else DEFAULT_PRICING_FILE
    # 상대 경로로 주었을 때 tracker 디렉토리의 pricing.yaml로 연결
    if not yaml_path.is_absolute():
        yaml_path = BASE_DIR / yaml_path
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data