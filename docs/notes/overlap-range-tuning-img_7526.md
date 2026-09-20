# Overlap-range tuning notes: `videos/IMG_7526.MOV`

Local dev notes from manually exercising `--mode overlap-greedy` (phase 4)
against a real iPhone clip kept in the gitignored `videos/` folder, not a
committed fixture. Recorded here so the reasoning and numbers don't have to
be re-derived next time `--overlap` needs retuning for a similar clip.

Clip: `videos/IMG_7526.MOV` -- 1080x1920 (portrait, rotation 90 from
container metadata), 29.97 fps, 2013 frames (~67s), HDR (HLG) tone-mapped,
decoded via the ffmpeg-subprocess backend.

## The question

A first run with the tool's default `--overlap 70-80` selected 1811 of 2013
analyzed frames (90%), with 752 of those via `fallback-closest` rather than
landing in-range. That looked suspicious -- almost every frame kept looks
like a bug (e.g. the overlap metric always reading low).

## Diagnosis: real motion, not a bug

Before retuning anything, checked whether the low overlap between adjacent
frames was genuine by measuring `orb_overlap` directly against decoded
frames, bypassing `select.py` entirely:

- Identical frame vs. itself (sanity check for a metric floor bug): `1.0`
  exactly, as expected -- no artificial floor.
- Literally adjacent frames (1/30s apart) at analysis resolution
  (270x480, `analysis_scale=0.25`): overlap starts at `0.83` and drifts
  down to `~0.77` within the first 10 frames.
- The same adjacent pair at full resolution (1080x1920): `0.655` --
  *lower* than at analysis resolution, ruling out "downscaling destroys
  ORB matches" as the cause.

Conclusion: this clip's frame-to-frame apparent motion is genuinely fast
enough that consecutive frames already sit in the 65-85% overlap band at
native 30fps. That's typical of a close-up orbit capture (small physical
motion still produces large image-plane displacement when the subject is
close). `--overlap 70-80` being fully default-appropriate for *smoother*
captures doesn't mean it's appropriate for this one -- the tool was doing
exactly what the spec asks (never silently leave a coverage gap), it just
had almost no room to skip frames without dropping below 70%.

## Runs

All runs: `uv run gs-frames extract videos/IMG_7526.MOV out --overlap <range>`
(default `--mode overlap-greedy --overlap-metric orb`, all other flags
default). "gap" = frame-index distance between consecutive selections.

| `--overlap` | selected / analyzed | in-range | fallback | mean gap | mean overlap |
|---|---|---|---|---|---|
| `70-80` (default) | 1811 / 2013 (90%) | 1058 | 752 | ~1.1 | 0.70 |
| `50-70` | 750 / 2013 (37%) | 744 | 5 | ~2.7 | 0.60 |
| `35-55` | 342 / 2013 (17%) | 336 | 5 | ~5.9 (max 48) | 0.45 |

The fallback count is the most telling number here: at `70-80` the selector
is constantly forced to fall back because nothing stays in-range long
enough; at `50-70` and `35-55` fallback drops to near zero (5/750, 5/342),
meaning the target range now actually matches how fast this footage's
overlap decays.

## Takeaway

- A high fallback-closest count relative to in-range is the signal to look
  at first when a selection looks too dense -- it means the requested
  overlap band is above what the footage's real motion allows, not
  necessarily a bug.
- Before assuming a metric bug, check the metric against real decoded
  frames directly (identical-frame sanity check + literal adjacent-frame
  overlap at both analysis and full resolution) rather than only looking at
  selection-level symptoms.
- For this specific clip (fast close-up orbit, ~30fps), `35-55` produces a
  well-spaced ~340-frame dataset; `50-70` is a middle ground (~750 frames)
  if denser coverage is wanted. `70-80` is not a good fit for this capture
  style.
- This is footage-specific, not a new default recommendation -- smoother
  captures (slower pans, farther subjects) may still overlap 90%+ at 30fps
  and want the `70-80` default or higher.
