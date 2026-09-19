from pathlib import Path

import numpy as np
import pytest

from gs_frames.export import export_images, image_path
from gs_frames.types import ConfigError, Selection


def _selection(index: int) -> Selection:
    return Selection(
        index=index,
        timestamp_s=index * 0.1,
        sharpness=1.0,
        overlap_with_prev=None,
        overlap_metric="none",
        reason="time-window",
    )


def _frame_source(indices):
    for i in sorted(indices):
        yield i, np.full((16, 16, 3), fill_value=i % 256, dtype=np.uint8)


def test_image_path_zero_pads_index_to_six_digits():
    assert image_path(Path("out"), 42, "jpg") == Path("out/frame_000042.jpg")
    assert image_path(Path("out"), 7, "png") == Path("out/frame_000007.png")


def test_export_writes_one_file_per_selection(tmp_path):
    out = tmp_path / "images"
    selections = [_selection(0), _selection(5), _selection(29)]
    written = export_images(
        _frame_source, selections, out, image_format="jpg", jpeg_quality=90, force=False
    )
    assert len(written) == 3
    for path in written:
        assert path.exists()
    assert sorted(p.name for p in written) == [
        "frame_000000.jpg",
        "frame_000005.jpg",
        "frame_000029.jpg",
    ]


def test_export_calls_on_frame_callback_per_written_frame(tmp_path):
    out = tmp_path / "images"
    selections = [_selection(0), _selection(1)]
    seen: list[int] = []
    export_images(
        _frame_source,
        selections,
        out,
        image_format="png",
        jpeg_quality=95,
        force=False,
        on_frame=seen.append,
    )
    assert seen == [0, 1]


def test_export_raises_without_force_when_files_already_exist(tmp_path):
    out = tmp_path / "images"
    selections = [_selection(0)]
    export_images(_frame_source, selections, out, image_format="jpg", jpeg_quality=90, force=False)

    with pytest.raises(ConfigError):
        export_images(
            _frame_source, selections, out, image_format="jpg", jpeg_quality=90, force=False
        )


def test_export_overwrites_with_force(tmp_path):
    out = tmp_path / "images"
    selections = [_selection(0)]
    export_images(_frame_source, selections, out, image_format="jpg", jpeg_quality=90, force=False)

    # Should not raise, and should still produce the file.
    written = export_images(
        _frame_source, selections, out, image_format="jpg", jpeg_quality=90, force=True
    )
    assert len(written) == 1
    assert written[0].exists()


def test_export_no_files_written_when_collision_detected(tmp_path):
    out = tmp_path / "images"
    selections = [_selection(0), _selection(1)]
    export_images(
        _frame_source, [_selection(0)], out, image_format="jpg", jpeg_quality=90, force=False
    )

    with pytest.raises(ConfigError):
        export_images(
            _frame_source, selections, out, image_format="jpg", jpeg_quality=90, force=False
        )
    # frame_000001 must not have been written by the failed call.
    assert not (out / "frame_000001.jpg").exists()
