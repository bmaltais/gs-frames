import csv
import json

from typer.testing import CliRunner

from gs_frames.cli import app

runner = CliRunner()


def test_help_works():
    result = runner.invoke(app, ["extract", "--help"])
    assert result.exit_code == 0
    assert "overlap" in result.output


def test_preview_writes_csv_row_per_analyzed_frame(synthetic_video, tmp_path):
    out = tmp_path / "out"
    result = runner.invoke(app, ["extract", str(synthetic_video), str(out), "--preview"])
    assert result.exit_code == 0, result.output

    csv_path = out / "analysis.csv"
    assert csv_path.exists()
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 30
    assert all(row["selected"] == "0" for row in rows)


def test_preview_writes_no_images(synthetic_video, tmp_path):
    out = tmp_path / "out"
    runner.invoke(app, ["extract", str(synthetic_video), str(out), "--preview"])
    assert not (out / "images").exists()


def test_without_preview_still_writes_csv_and_exits_zero(synthetic_video, tmp_path):
    out = tmp_path / "out"
    result = runner.invoke(app, ["extract", str(synthetic_video), str(out)])
    assert result.exit_code == 0, result.output
    assert "phase 2" in result.output
    assert (out / "analysis.csv").exists()
    assert not (out / "images").exists()


def test_manifest_records_rotation_and_source(synthetic_video, tmp_path):
    out = tmp_path / "out"
    runner.invoke(app, ["extract", str(synthetic_video), str(out), "--preview"])
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["video_meta"]["rotation_applied"] == 0
    assert manifest["video_meta"]["rotation_source"] in {
        "container_meta",
        "ffprobe",
        "user_override",
        "none_detected",
    }
    assert manifest["stats"]["analyzed"] == 30


def test_missing_video_is_user_error(tmp_path):
    result = runner.invoke(app, ["extract", str(tmp_path / "nope.mp4"), str(tmp_path / "out")])
    assert result.exit_code == 1


def test_invalid_overlap_range_is_user_error(synthetic_video, tmp_path):
    result = runner.invoke(
        app, ["extract", str(synthetic_video), str(tmp_path / "out"), "--overlap", "80-70"]
    )
    assert result.exit_code == 1
