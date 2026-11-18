from dataclasses import replace

from factor_zoo.config import load_settings
from factor_zoo.pipeline import run_demo


def test_demo_pipeline_runs_end_to_end(tmp_path) -> None:
    settings = load_settings("configs/base.yaml")
    settings = replace(
        settings,
        root=tmp_path,
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        results_dir=tmp_path / "results",
        figures_dir=tmp_path / "figures",
    )
    outputs = run_demo(settings, n_stocks=40, n_days=800)
    assert not outputs["factors_neutral"].empty
    assert not outputs["ic_summary"].empty
    assert "selected_factors" in outputs
