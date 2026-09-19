import csv
import dataclasses
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress

from gs_frames import __version__
from gs_frames import decode, sharpness
from gs_frames.types import ConfigError, ExtractConfig, FrameScore, RotationMode, SharpnessMethod

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
) -> None:
    config_dict = dataclasses.asdict(config)
    config_dict["video"] = str(config.video)
    config_dict["output_dir"] = str(config.output_dir)

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
        },
        "config": config_dict,
        "stats": {
            "analyzed": analyzed,
            "selected": 0,
            "in_range": 0,
            "fallback": 0,
        },
        "frames": [],
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
) -> None:
    """Analyze VIDEO and (from phase 2 onward) export selected frames to OUTPUT_DIR."""
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
            analysis_scale=analysis_scale,
            analysis_max_width=analysis_max_width,
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
            frame_meta: list[tuple[int, float]] = []

            with Progress(console=console) as progress:
                task = progress.add_task("Analyzing frames", total=dv.info.frame_count)
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
        )
        for i, (idx, ts) in enumerate(frame_meta)
    ]

    _write_analysis_csv(config.output_dir / "analysis.csv", scores, selected=set())
    _write_manifest(
        config.output_dir / "manifest.json",
        config=config,
        video_info=dv.info,
        analyzed=len(scores),
    )

    if not config.preview:
        console.print("Image export lands in phase 2; wrote analysis.csv and manifest.json only.")

    console.print(
        f"frames analyzed: {len(scores)}\n"
        f"frames selected: 0\n"
        f"overlap in-range: 0\n"
        f"overlap fallback: 0\n"
        f"output: {config.output_dir / 'images'}"
    )


@app.command()
def version() -> None:
    """Print the installed gs-frames version."""
    console.print(__version__)


if __name__ == "__main__":
    app()
