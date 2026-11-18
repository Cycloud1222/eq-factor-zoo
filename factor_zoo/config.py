"""Configuration loading with explicit local paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml


@dataclass(frozen=True)
class Settings:
    root: Path
    raw_dir: Path
    processed_dir: Path
    results_dir: Path
    figures_dir: Path
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    project_name: str
    data: dict

    @property
    def universe(self) -> dict:
        return self.data["universe"]

    @property
    def factors(self) -> dict:
        return self.data["factors"]

    @property
    def walk_forward(self) -> dict:
        return self.data["walk_forward"]

    @property
    def portfolio(self) -> dict:
        return self.data["portfolio"]

    def prepare_dirs(self) -> None:
        for path in (self.raw_dir, self.processed_dir, self.results_dir, self.figures_dir):
            path.mkdir(parents=True, exist_ok=True)


def load_settings(path: str | Path = "configs/base.yaml") -> Settings:
    config_path = Path(path).resolve()
    root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    project = raw["project"]
    paths = raw["paths"]
    return Settings(
        root=root,
        raw_dir=root / paths["raw_dir"],
        processed_dir=root / paths["processed_dir"],
        results_dir=root / paths["results_dir"],
        figures_dir=root / paths["figures_dir"],
        start_date=pd.Timestamp(project["start_date"]),
        end_date=pd.Timestamp(project["end_date"]),
        project_name=project["name"],
        data=raw,
    )
