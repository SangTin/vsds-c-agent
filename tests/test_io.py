from pathlib import Path

from src.io_utils import discover_input, write_predictions
from src.schema import Prediction


def test_discover_input_prefers_private_json(tmp_path: Path) -> None:
    public_json = tmp_path / "public-test_123.json"
    public_csv = tmp_path / "public_test.csv"
    private_csv = tmp_path / "private_test.csv"
    private_json = tmp_path / "private-test_456.json"
    for path in (public_json, public_csv, private_csv, private_json):
        path.touch()

    assert discover_input(tmp_path) == private_json


def test_discover_input_prefers_public_json_over_csv(tmp_path: Path) -> None:
    public_csv = tmp_path / "public_test.csv"
    public_json = tmp_path / "public-test_123.json"
    public_csv.touch()
    public_json.touch()

    assert discover_input(tmp_path) == public_json


def test_discover_input_prefers_private_csv_over_public_json(tmp_path: Path) -> None:
    public_json = tmp_path / "public_test.json"
    private_csv = tmp_path / "private-test_123.csv"
    public_json.touch()
    private_csv.touch()

    assert discover_input(tmp_path) == private_csv


def test_write_predictions_exact_bytes(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "pred.csv"
    predictions = [Prediction("q1", "A"), Prediction("q2", "B")]

    write_predictions(predictions, path)

    assert path.read_bytes() == b"qid,answer\nq1,A\nq2,B\n"
