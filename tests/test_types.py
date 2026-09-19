from pathlib import Path

import pytest

from gs_frames.types import ConfigError, ExtractConfig


def _config(**overrides) -> ExtractConfig:
    defaults = dict(video=Path(__file__), output_dir=Path("/tmp/unused"))
    defaults.update(overrides)
    return ExtractConfig(**defaults)


def test_force_defaults_to_false():
    assert _config().force is False


def test_chunk_frames_and_every_seconds_are_mutually_exclusive():
    with pytest.raises(ConfigError):
        _config(chunk_frames=10, every_seconds=1.0)


def test_chunk_frames_must_be_positive():
    with pytest.raises(ConfigError):
        _config(chunk_frames=0)


def test_every_seconds_must_be_positive():
    with pytest.raises(ConfigError):
        _config(every_seconds=-1.0)


def test_mode_defaults_to_time():
    assert _config().mode == "time"
