import tomllib
from pathlib import Path

import agent_convo


def test_runtime_version_matches_project_metadata() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text())

    assert agent_convo.__version__ == metadata["project"]["version"]
