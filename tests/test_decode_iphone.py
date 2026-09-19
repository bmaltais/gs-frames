"""Exercises decode.py's 8a handling (rotation, HEVC codec, VFR timestamps)
against a real iPhone clip. Skips with a clear reason if no fixture is
committed -- a synthetic CFR/unrotated/mp4v video does not substitute for
this, since it is always CFR, always right-side-up, and always H.264/mp4v.
"""

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
