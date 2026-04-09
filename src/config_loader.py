import yaml
from pathlib import Path
from datetime import datetime
from typing import Any, Dict


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Add run metadata
    now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    cfg["run_id"] = f"{cfg.get('experiment_name', 'exp')}_{now}"
    cfg["timestamp_utc"] = now

    # Ensure some paths are Path objects
    data_cfg = cfg.get("data", {})
    if "pdf_dir" in data_cfg:
        data_cfg["pdf_dir"] = str(Path(data_cfg["pdf_dir"]))
    if "corpus_cache" in data_cfg:
        data_cfg["corpus_cache"] = str(Path(data_cfg["corpus_cache"]))

    vector_cfg = cfg.get("vector_store", {})
    if "persist_dir" in vector_cfg:
        vector_cfg["persist_dir"] = str(Path(vector_cfg["persist_dir"]))

    return cfg


if __name__ == "__main__":
    # quick manual test
    c = load_config("config/base.yaml")
    print(c)
