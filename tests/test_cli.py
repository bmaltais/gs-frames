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
    selected = [row for row in rows if row["selected"] == "1"]
    assert 0 < len(selected) < 30  # overlap-greedy selection picks some, not all


def test_preview_writes_no_images(synthetic_video, tmp_path):
    out = tmp_path / "out"
    runner.invoke(app, ["extract", str(synthetic_video), str(out), "--preview"])
    assert not (out / "images").exists()


def test_without_preview_exports_images_matching_selected_count(synthetic_video, tmp_path):
    out = tmp_path / "out"
    result = runner.invoke(app, ["extract", str(synthetic_video), str(out)])
    assert result.exit_code == 0, result.output
    assert (out / "analysis.csv").exists()

    with (out / "analysis.csv").open() as f:
        rows = list(csv.DictReader(f))
    selected_indices = {int(row["index"]) for row in rows if row["selected"] == "1"}
    assert selected_indices

    images = sorted((out / "images").glob("frame_*.jpg"))
    assert {int(p.stem.split("_")[1]) for p in images} == selected_indices


def test_without_preview_requires_force_to_overwrite_existing_images(synthetic_video, tmp_path):
    out = tmp_path / "out"
    first = runner.invoke(app, ["extract", str(synthetic_video), str(out)])
    assert first.exit_code == 0, first.output

    second = runner.invoke(app, ["extract", str(synthetic_video), str(out)])
    assert second.exit_code == 1

    third = runner.invoke(app, ["extract", str(synthetic_video), str(out), "--force"])
    assert third.exit_code == 0, third.output


def test_unimplemented_mode_is_a_user_error(synthetic_video, tmp_path):
    result = runner.invoke(
        app, ["extract", str(synthetic_video), str(tmp_path / "out"), "--mode", "overlap-beam"]
    )
    assert result.exit_code == 1


def test_mode_overlap_greedy_is_now_the_default(synthetic_video, tmp_path):
    out = tmp_path / "out"
    result = runner.invoke(app, ["extract", str(synthetic_video), str(out), "--preview"])
    assert result.exit_code == 0, result.output
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["config"]["mode"] == "overlap-greedy"
    assert manifest["frames"][0]["reason"] == "seed"


def test_mode_overlap_greedy_exports_images_matching_selected_count(synthetic_video, tmp_path):
    out = tmp_path / "out"
    result = runner.invoke(
        app, ["extract", str(synthetic_video), str(out), "--mode", "overlap-greedy"]
    )
    assert result.exit_code == 0, result.output

    with (out / "analysis.csv").open() as f:
        rows = list(csv.DictReader(f))
    selected_indices = {int(row["index"]) for row in rows if row["selected"] == "1"}
    assert selected_indices

    images = sorted((out / "images").glob("frame_*.jpg"))
    assert {int(p.stem.split("_")[1]) for p in images} == selected_indices


def test_mode_overlap_greedy_with_flow_metric_needs_no_orb(synthetic_video, tmp_path):
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "extract",
            str(synthetic_video),
            str(out),
            "--mode",
            "overlap-greedy",
            "--overlap-metric",
            "flow",
            "--preview",
        ],
    )
    assert result.exit_code == 0, result.output
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["stats"]["selected"] > 0


def test_mode_time_exports_images_matching_selected_count(synthetic_video, tmp_path):
    out = tmp_path / "out"
    result = runner.invoke(app, ["extract", str(synthetic_video), str(out), "--mode", "time"])
    assert result.exit_code == 0, result.output

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["frames"][0]["reason"] == "time-window"


def test_mode_flow_exports_images_matching_selected_count(synthetic_video, tmp_path):
    out = tmp_path / "out"
    result = runner.invoke(
        app, ["extract", str(synthetic_video), str(out), "--mode", "flow", "--flow-trigger", "0.5"]
    )
    assert result.exit_code == 0, result.output

    with (out / "analysis.csv").open() as f:
        rows = list(csv.DictReader(f))
    selected_indices = {int(row["index"]) for row in rows if row["selected"] == "1"}
    assert selected_indices

    images = sorted((out / "images").glob("frame_*.jpg"))
    assert {int(p.stem.split("_")[1]) for p in images} == selected_indices

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["frames"][0]["reason"] == "flow-window"


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
    assert manifest["stats"]["selected"] > 0
    assert len(manifest["frames"]) == manifest["stats"]["selected"]
    assert manifest["frames"][0]["reason"] == "seed"


def test_missing_video_is_user_error(tmp_path):
    result = runner.invoke(app, ["extract", str(tmp_path / "nope.mp4"), str(tmp_path / "out")])
    assert result.exit_code == 1


def test_invalid_overlap_range_is_user_error(synthetic_video, tmp_path):
    result = runner.invoke(
        app, ["extract", str(synthetic_video), str(tmp_path / "out"), "--overlap", "80-70"]
    )
    assert result.exit_code == 1
