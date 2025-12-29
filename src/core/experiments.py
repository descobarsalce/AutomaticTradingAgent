import hashlib
import json
import os
import subprocess
from dataclasses import src.dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


def _safe_git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.getcwd()).decode().strip()
    except Exception:
        return "unknown"


def hash_state_dict(state_dict: Dict[str, Any]) -> str:
    """Compute a deterministic hash for a PyTorch state dict."""
    hasher = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        value = state_dict[key]
        hasher.update(key.encode())
        if hasattr(value, "cpu"):
            tensor = value.detach().cpu().numpy()
            hasher.update(tensor.tobytes())
        else:
            hasher.update(str(value).encode())
    return hasher.hexdigest()


@dataclass
class ExperimentManifest:
    run_id: str
    timestamp: str
    git_hash: str
    env_params: Dict[str, Any]
    ppo_params: Dict[str, Any]
    feature_config: Optional[Dict[str, Any]]
    data_range: Dict[str, Any]
    model_path: str
    state_dict_hash: str
    artifacts: List[str] = field(default_factory=list)
    deterministic_eval: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class ExperimentRegistry:
    def __init__(self, registry_path: str = "artifacts/experiments_log.jsonl") -> None:
        self.registry_path = registry_path
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)

    def register_run(self, manifest: ExperimentManifest) -> None:
        entry = manifest.to_dict()
        with open(self.registry_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    @staticmethod
    def save_manifest(manifest: ExperimentManifest, path: str) -> None:
        manifest_dir = os.path.dirname(path)
        if manifest_dir:
            os.makedirs(manifest_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)

    @staticmethod
    def load_manifest(path: str) -> ExperimentManifest:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ExperimentManifest(**data)

    @staticmethod
    def build_manifest(run_id: str,
                       env_params: Dict[str, Any],
                       ppo_params: Dict[str, Any],
                       feature_config: Optional[Dict[str, Any]],
                       data_range: Dict[str, Any],
                       model_path: str,
                       state_dict_hash: str,
                       artifacts: Optional[List[str]] = None,
                       deterministic_eval: bool = True) -> ExperimentManifest:
        return ExperimentManifest(
            run_id=run_id,
            timestamp=datetime.utcnow().isoformat(),
            git_hash=_safe_git_hash(),
            env_params=env_params,
            ppo_params=ppo_params,
            feature_config=feature_config,
            data_range=data_range,
            model_path=model_path,
            state_dict_hash=state_dict_hash,
            artifacts=artifacts or [],
            deterministic_eval=deterministic_eval,
        )
