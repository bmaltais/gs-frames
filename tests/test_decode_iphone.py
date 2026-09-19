"""Exercises decode.py's 8a handling (rotation, HEVC codec, VFR timestamps)
against a real iPhone clip. Skips with a clear reason if no fixture is
committed -- a synthetic CFR/unrotated/mp4v video does not substitute for
this, since it is always CFR, always right-side-up, and always H.264/mp4v.
"""

import cv2
import numpy as np
import pytest

from gs_frames import decode


# Bounds test runtime regardless of how long a real local clip happens to be
# (the spec's own fixture guidance expects ~3-5s; a dev's videos/ clip may be
# much longer). A few seconds is enough to exercise rotation/VFR handling.
_SAMPLE_SECONDS = 3.0


def test_iphone_fixture_decodes_portrait_with_rotation_applied(iphone_fixture):
    with decode.open_video(iphone_fixture) as dv:
        assert dv.info.width < dv.info.height, (
            "expected portrait output after rotation; a sideways landscape frame "
            "means rotation metadata was not applied"
        )
        frames = list(
            dv.iter_analysis_frames(
                analysis_scale=0.25, analysis_max_width=640, end_s=_SAMPLE_SECONDS
            )
        )
    assert len(frames) >= 10
    for af in frames:
        h, w = af.frame.shape[:2]
        assert w < h


def test_iphone_fixture_timestamps_are_non_decreasing(iphone_fixture):
    with decode.open_video(iphone_fixture) as dv:
        frames = list(
            dv.iter_analysis_frames(
                analysis_scale=0.25, analysis_max_width=640, end_s=_SAMPLE_SECONDS
            )
        )
    timestamps = [af.timestamp_s for af in frames]
    assert timestamps == sorted(timestamps)

    intervals = {round(b - a, 3) for a, b in zip(timestamps, timestamps[1:])}
    if len(intervals) <= 1:
        # Fixture happens to be CFR: this becomes a smoke test rather than a
        # VFR-specific assertion, as permitted by the spec.
        assert len(intervals) == 1


def test_ffmpeg_backend_frame_content_matches_opencv_backend(iphone_fixture):
    """Regression test for a real corruption bug: the ffmpeg-subprocess
    backend's raw-pipe reshape used the wrong width/height whenever ffmpeg's
    default autorotate pre-rotated its output frames out from under the
    probed (un-rotated) dimensions, producing scrambled/torn images that
    still happened to have "correct" width/height after our own rotation
    step -- so shape-only assertions (as in the tests above) never caught it.
    Cross-checking actual frame *content* between the two backends for the
    same source is what would have caught it.
    """
    from gs_frames.decode import _FfmpegBackend, _OpenCvBackend

    cv2_backend = _OpenCvBackend(iphone_fixture, rotation=0, use_container_timestamps=False)
    if not cv2_backend.usable():
        pytest.skip("this fixture isn't decodable via the OpenCV backend; nothing to compare")
    _, cv2_frame_raw = cv2_backend.cap.read()
    cv2_backend.close()
    assert cv2_frame_raw is not None

    ffmpeg_backend = _FfmpegBackend(iphone_fixture, rotation=0, tonemap=False)
    ffmpeg_frame_raw = next(iter(ffmpeg_backend.iter_frames()))[2]

    assert cv2_frame_raw.shape == ffmpeg_frame_raw.shape, (
        "both backends must decode the same source to the same raw (un-rotated) "
        "frame shape, or downstream rotation/reshape math will silently disagree"
    )

    def _small_gray(frame):
        small = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float64).ravel()

    a, b = _small_gray(cv2_frame_raw), _small_gray(ffmpeg_frame_raw)
    correlation = np.corrcoef(a, b)[0, 1]
    assert correlation > 0.7, (
        f"ffmpeg backend's first frame is not structurally similar to the OpenCV "
        f"backend's first frame of the same source (correlation={correlation:.2f}); "
        "likely a reshape/orientation mismatch between the two decode paths"
    )


def test_iphone_fixture_hdr_frames_are_tonemapped(iphone_fixture):
    """Apple's "HDR Video" capture mode (10-bit HEVC, HLG/Rec.2020) reads out
    far too bright if decoded without tone-mapping, since cv2 applies no HDR
    EOTF of its own. Only meaningful if the local fixture happens to be HDR;
    an SDR fixture makes this a no-op smoke assertion instead.
    """
    with decode.open_video(iphone_fixture) as dv:
        if not dv.info.hdr_tonemapped:
            pytest.skip("local iPhone fixture is not HDR; nothing to verify here")
        frames = list(
            dv.iter_analysis_frames(
                analysis_scale=0.25, analysis_max_width=640, end_s=_SAMPLE_SECONDS
            )
        )
    assert frames
    # Generous bound, not exact equality (tonemap filter output varies by
    # ffmpeg version): a properly tone-mapped SDR frame from typical daylight
    # footage shouldn't be pegged near white.
    assert all(af.frame.mean() < 235 for af in frames)
