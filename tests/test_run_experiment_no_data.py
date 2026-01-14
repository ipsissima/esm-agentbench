import pytest
from pathlib import Path

from analysis.run_experiment import run_experiment


def test_run_experiment_fails_on_no_traces(tmp_path: Path):
    traces_dir = tmp_path / "experiment_traces"
    traces_dir.mkdir()
    out_dir = tmp_path / "reports"
    out_dir.mkdir()
    with pytest.raises(SystemExit):
        run_experiment(traces_dir, out_dir, k=10, scenario_name='test_scenario')
