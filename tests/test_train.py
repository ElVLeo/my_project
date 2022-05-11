from click.testing import CliRunner
import pytest
from forest_ml.train import train


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def test_error_for_criterion(runner: CliRunner) -> None:
    """It fails when criterion is not real."""
    result = runner.invoke(
        train,
        [
            "--criterion",
            "gin",
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--criterion'" in result.output


def test_error_for_grid_search(runner: CliRunner) -> None:
    """It is valid when grid_search is False."""
    result = runner.invoke(
        train,
        [
            "--grid_search",
            "False",
        ],
    )
    assert result.exit_code == 0
