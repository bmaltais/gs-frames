import csv
import dataclasses
import json
import logging
import sys
from pathlib import Path
from typing import Optional, cast

import numpy as np
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from gs_frames import __version__
from gs_frames import decode, export, motion, select, sharpness
from gs_frames.types import (
    ConfigError,
    ExtractConfig,
    FrameScore,
    ImageFormat,
    RotationMode,
    Selection,
    SelectMode,
    SharpnessMethod,
)

app = typer.Typer(add_completion=False)
console = Console()


def _parse_overlap(overlap: str) -> tuple[float, float]:
    try:
        lo_str, hi_str = overlap.split("-", 1)
        lo, hi = float(lo_str), float(hi_str)
    except ValueError as exc:
        raise ConfigError(f"invalid --overlap value: {overlap!r}") from exc
    if lo > 1 or hi > 1:
        lo, hi = lo / 100, hi / 100
    return lo, hi


def _estimate_analysis_frame_count(
    config: ExtractConfig, video_info: decode.VideoInfo
) -> Optional[int]:
    """Best-effort frame count for the progress bar's total, honoring
    --start/--end bounds so a trimmed run doesn't show progress/ETA against
    the full video's frame count. Approximate under VFR (uses nominal fps);
    display-only, never used for selection logic.
    """
    start_frame = config.start_frame or 0
    if config.start_s is not None and video_info.fps:
        start_frame = max(start_frame, round(config.start_s * video_info.fps))

    end_frame: Optional[int]
    if config.end_frame is not None:
        end_frame = config.end_frame
    elif config.end_s is not None and video_info.fps:
        end_frame = round(config.end_s * video_info.fps)
    else:
        end_frame = video_info.frame_count

    if end_frame is None:
        return None
    return max(0, end_frame - start_frame)


def _selection_stats(selections: list[Selection]) -> tuple[int, int]:
    """(in_range, fallback) counts derived from selection reasons -- shared by
    the manifest and the final console summary so they can't drift apart."""
    in_range = sum(1 for s in selections if s.reason == "in-range")
    fallback = sum(1 for s in selections if s.reason == "fallback-closest")
    return in_range, fallback


def _rotation_mode_for(applied_degrees: int) -> RotationMode:
    """The explicit RotationMode that reproduces an already-resolved rotation,
    so re-opening the video for export doesn't re-run auto-detection."""
    if applied_degrees == 0:
        return "none"
    return cast(RotationMode, str(applied_degrees))


def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("gs_frames")
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(RichHandler(console=console, show_path=False))
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(file_handler)


def _write_analysis_csv(path: Path, scores: list[FrameScore], selected: set[int]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index",
                "timestamp_s",
                "sharpness",
                "sharpness_tenengrad",
                "sharpness_laplacian",
                "flow_median",
                "selected",
            ]
        )
        for s in scores:
            writer.writerow(
                [
                    s.index,
                    f"{s.timestamp_s:.6f}",
                    f"{s.sharpness:.6f}",
                    f"{s.sharpness_tenengrad:.6f}",
                    f"{s.sharpness_laplacian:.6f}",
                    f"{s.flow_median:.6f}",
                    int(s.index in selected),
                ]
            )


def _write_manifest(
    path: Path,
    *,
    config: ExtractConfig,
    video_info: decode.VideoInfo,
    analyzed: int,
    selections: list[Selection],
) -> None:
    config_dict = dataclasses.asdict(config)
    config_dict["video"] = str(config.video)
    config_dict["output_dir"] = str(config.output_dir)
    in_range, fallback = _selection_stats(selections)

    manifest = {
        "version": __version__,
        "video": str(config.video),
        "video_meta": {
            "fps": video_info.fps,
            "width": video_info.width,
            "height": video_info.height,
            "frame_count": video_info.frame_count,
            "rotation_applied": video_info.rotation_applied,
            "rotation_source": video_info.rotation_source,
            "hdr_tonemapped": video_info.hdr_tonemapped,
        },
        "config": config_dict,
        "stats": {
            "analyzed": analyzed,
            "selected": len(selections),
            "in_range": in_range,
            "fallback": fallback,
        },
        "frames": [dataclasses.asdict(s) for s in selections],
    }
    path.write_text(json.dumps(manifest, indent=2))


@app.command()
def extract(
    video: Path = typer.Argument(..., help="Input video file."),
    output_dir: Path = typer.Argument(..., help="Directory to write analysis/selection/images to."),
    overlap: str = typer.Option("70-80", "--overlap", help="Target overlap range, e.g. 70-80."),
    overlap_min: Optional[float] = typer.Option(None, "--overlap-min"),
    overlap_max: Optional[float] = typer.Option(None, "--overlap-max"),
    target_overlap: Optional[float] = typer.Option(None, "--target-overlap"),
    sharpness_method: SharpnessMethod = typer.Option("combined", "--sharpness"),
    analysis_scale: float = typer.Option(0.25, "--analysis-scale"),
    analysis_max_width: int = typer.Option(640, "--analysis-max-width"),
    preview: bool = typer.Option(False, "--preview"),
    start_seconds: Optional[float] = typer.Option(None, "--start-seconds"),
    end_seconds: Optional[float] = typer.Option(None, "--end-seconds"),
    start_frame: Optional[int] = typer.Option(None, "--start-frame"),
    end_frame: Optional[int] = typer.Option(None, "--end-frame"),
    workers: int = typer.Option(1, "--workers"),
    rotate: RotationMode = typer.Option("auto", "--rotate"),
    no_container_timestamps: bool = typer.Option(False, "--no-container-timestamps"),
    mode: SelectMode = typer.Option("time", "--mode", help="Selection mode."),
    chunk_frames: Optional[int] = typer.Option(
        None, "--chunk-frames", help="--mode time: pick the sharpest frame every N frames."
    ),
    every_seconds: Optional[float] = typer.Option(
        None, "--every-seconds", help="--mode time: pick the sharpest frame every N seconds."
    ),
    flow_trigger: float = typer.Option(
        8.0, "--flow-trigger", help="--mode flow: accumulated flow_median that closes a window."
    ),
    max_frames: Optional[int] = typer.Option(None, "--max-frames", help="Cap selected frames."),
    min_sharpness: Optional[float] = typer.Option(
        None, "--min-sharpness", help="Frames below this score are never selected."
    ),
    min_sharpness_percentile: float = typer.Option(
        5.0, "--min-sharpness-percentile", help="Used when --min-sharpness is unset."
    ),
    image_format: ImageFormat = typer.Option("jpg", "--format"),
    quality: int = typer.Option(95, "--quality", help="JPEG quality (1-100)."),
    force: bool = typer.Option(False, "--force", help="Overwrite existing exported images."),
) -> None:
    """Analyze VIDEO and export selected frames to OUTPUT_DIR."""
    if mode not in select.IMPLEMENTED_MODES:
        console.print(
            f"[red]error:[/red] --mode {mode} is not implemented yet; "
            f"this build supports only {sorted(select.IMPLEMENTED_MODES)} "
            "(overlap-greedy becomes the default in phase 4)."
        )
        raise typer.Exit(1)

    lo, hi = _parse_overlap(overlap)
    if overlap_min is not None or overlap_max is not None:
        if overlap != "70-80":
            console.print("[yellow]warning:[/yellow] --overlap-min/--overlap-max override --overlap")
        lo = overlap_min if overlap_min is not None else lo
        hi = overlap_max if overlap_max is not None else hi

    try:
        config = ExtractConfig(
            video=video,
            output_dir=output_dir,
            overlap_min=lo,
            overlap_max=hi,
            target_overlap=target_overlap,
            sharpness=sharpness_method,
            mode=mode,
            analysis_scale=analysis_scale,
            analysis_max_width=analysis_max_width,
            chunk_frames=chunk_frames,
            every_seconds=every_seconds,
            flow_trigger=flow_trigger,
            max_frames=max_frames,
            min_sharpness=min_sharpness,
            min_sharpness_percentile=min_sharpness_percentile,
            format=image_format,
            jpeg_quality=quality,
            force=force,
            preview=preview,
            start_s=start_seconds,
            end_s=end_seconds,
            start_frame=start_frame,
            end_frame=end_frame,
            workers=workers,
            rotation=rotate,
            use_container_timestamps=not no_container_timestamps,
        )
    except ConfigError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(config.output_dir / "gs-frames.log")
    logger = logging.getLogger("gs_frames.cli")

    try:
        with decode.open_video(config.video, config.rotation, config.use_container_timestamps) as dv:
            logger.info("resolved overlap range: [%.2f, %.2f]", config.overlap_min, config.overlap_max)
            tenengrad_vals: list[float] = []
            laplacian_vals: list[float] = []
            flow_vals: list[float] = []
            frame_meta: list[tuple[int, float]] = []
            prev_gray: Optional[np.ndarray] = None

            progress_columns = (
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TextColumn("eta"),
                TimeRemainingColumn(),
            )
            with Progress(*progress_columns, console=console) as progress:
                total = _estimate_analysis_frame_count(config, dv.info)
                task = progress.add_task("Analyzing frames", total=total)
                for af in dv.iter_analysis_frames(
                    analysis_scale=config.analysis_scale,
                    analysis_max_width=config.analysis_max_width,
                    start_frame=config.start_frame or 0,
                    end_frame=config.end_frame,
                    start_s=config.start_s,
                    end_s=config.end_s,
                ):
                    gray = sharpness.to_gray(af.frame)
                    tenengrad_vals.append(sharpness.tenengrad(gray))
                    laplacian_vals.append(sharpness.laplacian_variance(gray))
                    flow_vals.append(
                        motion.flow_median(prev_gray, gray, cell_size=config.flow_cell_size)
                        if prev_gray is not None
                        else 0.0
                    )
                    prev_gray = gray
                    frame_meta.append((af.index, af.timestamp_s))
                    progress.advance(task)
    except (RuntimeError, OSError) as exc:
        console.print(f"[red]error:[/red] failed to decode video: {exc}")
        raise typer.Exit(2)

    if not frame_meta:
        console.print("[red]error:[/red] no frames decoded from video")
        raise typer.Exit(2)

    combined = sharpness.combined_scores(np.array(tenengrad_vals), np.array(laplacian_vals))
    method_values = {
        "tenengrad": tenengrad_vals,
        "laplacian": laplacian_vals,
        "combined": combined.tolist(),
    }
    chosen = method_values[config.sharpness]

    scores = [
        FrameScore(
            index=idx,
            timestamp_s=ts,
            sharpness=chosen[i],
            sharpness_tenengrad=tenengrad_vals[i],
            sharpness_laplacian=laplacian_vals[i],
            flow_median=flow_vals[i],
        )
        for i, (idx, ts) in enumerate(frame_meta)
    ]

    with console.status("Selecting frames..."):
        selections = select.select_frames(scores, config)

    _write_analysis_csv(
        config.output_dir / "analysis.csv", scores, selected={s.index for s in selections}
    )
    _write_manifest(
        config.output_dir / "manifest.json",
        config=config,
        video_info=dv.info,
        analyzed=len(scores),
        selections=selections,
    )

    if config.preview:
        console.print("[dim]--preview: skipping image export.[/dim]")
    elif selections:
        export_rotation = _rotation_mode_for(dv.info.rotation_applied)
        try:
            with decode.open_video(
                config.video, export_rotation, config.use_container_timestamps
            ) as export_dv:
                with Progress(*progress_columns, console=console) as progress:
                    task = progress.add_task("Exporting frames", total=len(selections))
                    export.export_images(
                        export_dv.iter_export_frames,
                        selections,
                        config.output_dir / "images",
                        image_format=config.format,
                        jpeg_quality=config.jpeg_quality,
                        force=config.force,
                        on_frame=lambda _index: progress.advance(task),
                    )
        except ConfigError as exc:
            console.print(f"[red]error:[/red] {exc}")
            raise typer.Exit(1)
        except (RuntimeError, OSError) as exc:
            console.print(f"[red]error:[/red] failed to export images: {exc}")
            raise typer.Exit(2)

    in_range, fallback = _selection_stats(selections)
    console.print(
        f"frames analyzed: {len(scores)}\n"
        f"frames selected: {len(selections)}\n"
        f"overlap in-range: {in_range}\n"
        f"overlap fallback: {fallback}\n"
        f"output: {config.output_dir / 'images'}"
    )


@app.command()
def version() -> None:
    """Print the installed gs-frames version."""
    console.print(__version__)


if __name__ == "__main__":
    app()
