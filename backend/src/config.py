from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    log_level: str = "info"


class ExtractiveConfig(BaseModel):
    embedding_model: str = "all-MiniLM-L6-v2"
    max_sentences: int = 50
    top_k_short: int = 3
    top_k_long: int = 7
    redundancy_lambda: float = 0.7
    position_bias: float = 1.1
    textrank_damping: float = 0.85


class AbstractiveConfig(BaseModel):
    model_name: str = "facebook/bart-large-cnn"
    device: str = "auto"
    max_input_length: int = 1024
    max_output_length: int = 150
    min_output_length: int = 60
    num_beams: int = 4
    length_penalty: float = 2.0
    no_repeat_ngram_size: int = 3
    early_stopping: bool = True


class PostProcessingConfig(BaseModel):
    trigram_block: bool = True
    target_length_ratio: float = 0.07
    repetition_threshold: float = 0.85


class DataConfig(BaseModel):
    cache_dir: Path = Path("./models/pretrained")
    nltk_data_dir: Path = Path("./data/nltk")


class Settings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    models: Dict[str, Any] = Field(default_factory=dict)
    postprocessing: PostProcessingConfig = Field(default_factory=PostProcessingConfig)
    evaluation: Dict[str, Any] = Field(default_factory=dict)
    data: DataConfig = Field(default_factory=DataConfig)

    extractive: ExtractiveConfig | None = None
    abstractive: AbstractiveConfig | None = None

    def model_post_init(self, __context: Optional[Dict[str, Any]]) -> None:
        model_cfg = self.models or {}
        if "extractive" in model_cfg:
            self.extractive = ExtractiveConfig(**model_cfg["extractive"])
        else:
            self.extractive = ExtractiveConfig()
        if "abstractive" in model_cfg:
            self.abstractive = AbstractiveConfig(**model_cfg["abstractive"])
        else:
            self.abstractive = AbstractiveConfig()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@lru_cache()
def get_settings(config_path: str | Path | None = None) -> Settings:
    if config_path is None:
        # Try multiple possible locations
        possible_paths = [
            Path("backend/config/config.yaml"),
            Path(__file__).parent.parent.parent / "config" / "config.yaml",
            Path(__file__).parent.parent / "config" / "config.yaml",
        ]
        path = None
        for p in possible_paths:
            if p.exists():
                path = p
                break
        if path is None:
            # Return default settings if no config file found
            return Settings()
    else:
        path = Path(config_path)
        if not path.exists():
            # Return default settings if specified path doesn't exist
            return Settings()
    
    raw = _load_yaml(path)
    return Settings(**raw)


