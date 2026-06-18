"""
scene_schema.py — the schema for an educational animation scene.

A Scene is an ordered list of SceneSteps. Crucially, **a step references a
formula by `formula_id` only** — it never carries raw LaTeX or paraphrased
formula text. This makes the ledger the single source of truth: the renderer
pulls the canonical LaTeX/label/symbols out of the ledger at render time, so a
scene can never drift from the reviewed formula.

Allowed step kinds (all reference a formula in the ledger):
  - "show_label"   : display the formula's human label (Formula.label)
  - "show_formula" : display the formula's canonical LaTeX (Formula.latex)
  - "show_legend"  : display a legend for the formula's symbols; an optional
                     `highlight_symbols` list restricts/emphasises symbol ids.

The only free-text in a scene is `Scene.title` (prose), which the validator
checks contains no LaTeX so it cannot smuggle in an un-ledgered formula.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

STEP_KINDS = ("show_label", "show_formula", "show_legend")


@dataclass(frozen=True)
class SceneStep:
    """One animation beat. References a formula by id; carries no raw text."""
    kind:               str
    formula_id:         str
    highlight_symbols:  Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind":              self.kind,
            "formula_id":        self.formula_id,
            "highlight_symbols": list(self.highlight_symbols),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SceneStep":
        return SceneStep(
            kind=d["kind"],
            formula_id=d["formula_id"],
            highlight_symbols=tuple(d.get("highlight_symbols", []) or ()),
        )


@dataclass(frozen=True)
class Scene:
    """An ordered storyboard of formula-referencing steps."""
    id:    str
    title: str
    steps: Tuple[SceneStep, ...]

    def referenced_formula_ids(self) -> List[str]:
        seen: List[str] = []
        for s in self.steps:
            if s.formula_id not in seen:
                seen.append(s.formula_id)
        return seen

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":    self.id,
            "title": self.title,
            "steps": [s.to_dict() for s in self.steps],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Scene":
        return Scene(
            id=d["id"],
            title=d.get("title", ""),
            steps=tuple(SceneStep.from_dict(s) for s in d.get("steps", [])),
        )


def build_area_triangle_scene() -> Scene:
    """
    Canonical worked-example scene for "Area of a Triangle".

    Every step points at the `area_triangle` formula id; not a single character
    of LaTeX is duplicated here.
    """
    fid = "area_triangle"
    return Scene(
        id="area_triangle_worked_example",
        title="Worked Example",
        steps=(
            SceneStep("show_label",   fid),
            SceneStep("show_formula", fid),
            SceneStep("show_legend",  fid),
        ),
    )


__all__ = ["STEP_KINDS", "SceneStep", "Scene", "build_area_triangle_scene"]
