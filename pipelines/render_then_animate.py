"""
render_then_animate.py — render canonical formulas, then animate a scene.

Workflow (deterministic, local, no network):
  1. VALIDATE  — run the full formula_validator over the ledger and the scene.
                 Abort with a non-zero exit code on ANY issue. Nothing is ever
                 rendered from an unvalidated/incorrect formula.
  2. RENDER    — for every formula referenced by the scene, fetch its canonical
                 LaTeX from the ledger *verbatim* and typeset it to a PNG via
                 matplotlib mathtext. No heuristic rewriting happens here.
  3. ANIMATE   — replay the scene steps as a cumulative reveal: each frame adds
                 the next step (label -> formula -> symbol legend). Frames are
                 stitched into a GIF with Pillow.
  4. MANIFEST  — write a JSON audit trail recording the exact LaTeX (and its
                 SHA-256) pulled from the ledger for each rendered formula.

Outputs (default): assets/pedagogy/<scene_id>/
  render/<formula_id>.png   canonical static renders
  frames/frame_XX.png       cumulative animation frames
  <scene_id>.gif            the animation
  manifest.json             audit trail

The worked example is "Area of a Triangle" (formula id: area_triangle).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from pedagogy.formula_ledger import Formula, FormulaLedger, load_ledger
from pedagogy.scene_schema import Scene, SceneStep, build_area_triangle_scene
from pedagogy.formula_validator import validate_all

# Deterministic rendering knobs (no timestamps, fixed geometry/fonts).
FIG_W, FIG_H, DPI = 8.0, 4.5, 100
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"


def _blank_ax():
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def _legend_lines(formula: Formula, highlight) -> List[str]:
    """Build legend rows straight from the ledger symbols (exact tokens)."""
    rows = []
    hl = set(highlight or ())
    for s in formula.symbols:
        if hl and s.id not in hl:
            continue
        unit = f"  [{s.unit}]" if s.unit else ""
        mark = "\u25b6 " if (hl and s.id in hl) else ""
        rows.append((s.token, f"{mark}{s.name}: {s.description}{unit}"))
    return rows


# ── stage 2: render a single canonical formula ───────────────────────────────
def render_formula_png(formula: Formula, path: str) -> None:
    fig, ax = _blank_ax()
    ax.text(0.5, 0.5, f"${formula.latex}$", ha="center", va="center", fontsize=40)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ── stage 3: one cumulative animation frame ──────────────────────────────────
def render_frame(scene: Scene, ledger: FormulaLedger, upto: int, path: str) -> None:
    fig, ax = _blank_ax()
    ax.text(0.5, 0.95, scene.title, ha="center", va="top", fontsize=16, color="#444444")

    y = 0.78
    for step in scene.steps[:upto + 1]:
        formula = ledger.get(step.formula_id)
        if step.kind == "show_label":
            ax.text(0.5, y, formula.label, ha="center", va="center",
                    fontsize=24, color="black")
            y -= 0.14
        elif step.kind == "show_formula":
            ax.text(0.5, y, f"${formula.latex}$", ha="center", va="center",
                    fontsize=40, color="black")
            y -= 0.20
        elif step.kind == "show_legend":
            for token, desc in _legend_lines(formula, step.highlight_symbols):
                ax.text(0.12, y, f"${token}$", ha="left", va="center", fontsize=18)
                ax.text(0.20, y, desc, ha="left", va="center", fontsize=14,
                        color="#222222")
                y -= 0.09
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run(out_root: str = "assets/pedagogy", duration_ms: int = 900) -> int:
    ledger = load_ledger()
    scene = build_area_triangle_scene()

    # ── stage 1: validate (hard gate) ────────────────────────────────────────
    issues = validate_all(ledger, [scene])
    if issues:
        print(f"VALIDATION FAILED — {len(issues)} issue(s); nothing rendered:")
        for iss in sorted(issues, key=lambda x: (x.where, x.code, x.message)):
            print("  " + str(iss))
        return 1
    print("Validation passed — rendering from the ledger verbatim.")

    out_dir = os.path.join(out_root, scene.id)
    render_dir = os.path.join(out_dir, "render")
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(render_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    # ── stage 2: render each referenced formula verbatim ─────────────────────
    manifest = {"scene_id": scene.id, "title": scene.title, "formulas": []}
    for fid in scene.referenced_formula_ids():
        formula = ledger.get(fid)
        png = os.path.join(render_dir, f"{fid}.png")
        render_formula_png(formula, png)
        manifest["formulas"].append({
            "id": formula.id,
            "label": formula.label,
            "latex": formula.latex,             # exact, from ledger
            "latex_sha256": _sha256(formula.latex),
            "render": os.path.relpath(png, out_root),
            "symbols": [
                {"id": s.id, "token": s.token, "name": s.name,
                 "kind": s.kind, "unit": s.unit}
                for s in formula.symbols
            ],
        })
        print(f"  rendered formula {fid!r} -> {png}")

    # ── stage 3: cumulative animation frames + GIF ───────────────────────────
    frame_paths: List[str] = []
    for i in range(len(scene.steps)):
        fp = os.path.join(frames_dir, f"frame_{i:02d}.png")
        render_frame(scene, ledger, i, fp)
        frame_paths.append(fp)
    print(f"  rendered {len(frame_paths)} animation frame(s) -> {frames_dir}")

    gif_path = os.path.join(out_dir, f"{scene.id}.gif")
    frames = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in frame_paths]
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0, disposal=2)
    print(f"  animated GIF -> {gif_path}")

    # ── stage 4: manifest audit trail ────────────────────────────────────────
    manifest["frames"] = [os.path.relpath(p, out_root) for p in frame_paths]
    manifest["gif"] = os.path.relpath(gif_path, out_root)
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    print(f"  audit manifest -> {manifest_path}")

    print(f"\nDone. Worked example assets in: {out_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="assets/pedagogy",
                        help="output root directory (default: assets/pedagogy)")
    parser.add_argument("--duration", type=int, default=900,
                        help="per-frame GIF duration in ms (default: 900)")
    args = parser.parse_args()
    return run(args.out, args.duration)


if __name__ == "__main__":
    sys.exit(main())
