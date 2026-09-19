"""Exercises decode.py's output as observed by callers: analysis-resolution
frames and the full-resolution seek-with-fallback export path. Never asserts
on internal OpenCV call sequences.
"""

from gs_frames import decode


def test_iter_export_frames_returns_requested_indices_at_full_resolution(synthetic_video):
    with decode.open_video(synthetic_video) as dv:
        analysis_frames = list(
            dv.iter_analysis_frames(analysis_scale=0.25, analysis_max_width=640)
        )
        targets = [0, 5, 29]
        exported = list(dv.iter_export_frames(targets))

    assert [idx for idx, _ in exported] == targets
    for _, frame in exported:
        assert frame.shape[:2] == (240, 320)  # synthetic_video's native size

    analysis_h, analysis_w = analysis_frames[0].frame.shape[:2]
    assert (analysis_h, analysis_w) != (240, 320)  # analysis frames are downscaled


def test_iter_export_frames_handles_unsorted_and_duplicate_indices(synthetic_video):
    with decode.open_video(synthetic_video) as dv:
        exported = list(dv.iter_export_frames([10, 3, 3, 7]))
    assert [idx for idx, _ in exported] == [3, 7, 10]


def test_iter_export_frames_empty_indices_yields_nothing(synthetic_video):
    with decode.open_video(synthetic_video) as dv:
        assert list(dv.iter_export_frames([])) == []


def test_sdr_synthetic_video_is_not_tonemapped(synthetic_video):
    with decode.open_video(synthetic_video) as dv:
        assert dv.info.hdr_tonemapped is False
        assert dv.info.backend == "opencv"  # tonemap detection shouldn't force the ffmpeg path


# The seek-lands-inaccurately fallback branch in iter_export_frames is not
# covered here: cv2.VideoCapture's get()/set() are read-only C-extension
# attributes (no monkeypatching), and this project's mp4v synthetic fixture
# always seeks accurately in practice, so there is no way to force the
# failure signal without asserting on internal OpenCV call sequences, which
# the spec's testing philosophy for decode.py explicitly rules out.
