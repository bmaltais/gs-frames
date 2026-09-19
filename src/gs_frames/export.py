"""Full-resolution image export for selected frames.

Decoupled from decode.DecodedVideo via an injected frame_source callable, the
same pattern as select.py's overlap_fn, so export logic (filenames, overwrite
protection, format/quality) is testable without real video I/O.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence

import cv2
import numpy as np

from gs_frames.types import ConfigError, ImageFormat, Selection

FrameSource = Callable[[Sequence[int]], Iterator[tuple[int, np.ndarray]]]


def image_path(output_dir: Path, index: int, image_format: ImageFormat) -> Path:
    return output_dir / f"frame_{index:06d}.{image_format}"


def export_images(
    frame_source: FrameSource,
    selections: Sequence[Selection],
    output_dir: Path,
    *,
    image_format: ImageFormat,
    jpeg_quality: int,
    force: bool,
    on_frame: Optional[Callable[[int], None]] = None,
) -> list[Path]:
    """Writes one full-resolution image per selection to output_dir.

    Checks for existing target files up front and raises ConfigError before
    writing anything if any are found and force is False, so a rerun without
    --force never partially clobbers a prior export.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = [s.index for s in selections]
    paths = {idx: image_path(output_dir, idx, image_format) for idx in indices}

    if not force:
        existing = [p for p in paths.values() if p.exists()]
        if existing:
            raise ConfigError(
                f"{len(existing)} image(s) already exist in {output_dir} "
                f"(e.g. {existing[0].name}); pass --force to overwrite"
            )

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if image_format == "jpg" else []

    written: list[Path] = []
    for index, frame in frame_source(indices):
        path = paths[index]
        if not cv2.imwrite(str(path), frame, encode_params):
            raise RuntimeError(f"failed to write image: {path}")
        written.append(path)
        if on_frame is not None:
            on_frame(index)

    return written
