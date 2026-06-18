"""
formula_ledger.py — the single source of truth for every formula shown in an
educational animation.

CANONICAL FORMULA POLICY
------------------------
A formula is a *canonical text object*. Once it lives in the ledger its LaTeX,
label, symbols, subscripts/superscripts, Greek letters and units are immutable
and must be used verbatim. Downstream code (scenes, validators, renderers) may
only *reference* a formula by its `id` — it may never paraphrase, re-typeset, or
heuristically rewrite the LaTeX. This guarantees that what the learner sees is
exactly what was authored and reviewed.

The ledger is pure Python data (no I/O, no network) so the whole system stays
deterministic and local.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Symbol:
    """
    One canonical symbol used inside a formula.

    Fields:
      id          global concept id, shared across formulas (e.g. "side_b").
                  Two formulas that use the same concept MUST reuse the same id
                  and the same `token` — the validator enforces this, which is
                  how casing/symbol drift is detected.
      token       the EXACT LaTeX atom as it appears in the formula, e.g.
                  "b", "A", "\\pi", "\\Delta". Never paraphrased.
      name        short human name, e.g. "base".
      kind        one of: variable | constant | greek | operator.
      description short prose gloss (no LaTeX).
      unit        physical unit string (prose, e.g. "m", "s", "m/s",
                  "unit\u00b2"); empty string if dimensionless.
    """
    id:          str
    token:       str
    name:        str
    kind:        str
    description: str
    unit:        str = ""


@dataclass(frozen=True)
class Formula:
    """
    A canonical formula text object.

    Fields:
      id          stable key referenced by scenes (e.g. "area_triangle").
      latex       canonical LaTeX, used verbatim by the renderer.
      label       human title (prose, no LaTeX), e.g. "Area of a Triangle".
      description short prose explanation (no LaTeX).
      symbols     ordered tuple of Symbol objects appearing in `latex`.
      plaintext   optional ASCII fallback, also canonical/immutable.
    """
    id:          str
    latex:       str
    label:       str
    description: str
    symbols:     Tuple[Symbol, ...]
    plaintext:   str = ""


class FormulaLedger:
    """Immutable, ordered collection of canonical formulas keyed by id."""

    def __init__(self, formulas: List[Formula]):
        self._formulas: "OrderedDict[str, Formula]" = OrderedDict()
        for f in formulas:
            if f.id in self._formulas:
                raise ValueError(f"duplicate formula id in ledger: {f.id!r}")
            self._formulas[f.id] = f

    # ── read-only access ────────────────────────────────────────────────────
    def ids(self) -> List[str]:
        return list(self._formulas.keys())

    def formulas(self) -> List[Formula]:
        return list(self._formulas.values())

    def get(self, formula_id: str) -> Formula:
        if formula_id not in self._formulas:
            raise KeyError(f"unknown formula id: {formula_id!r}")
        return self._formulas[formula_id]

    def __contains__(self, formula_id: str) -> bool:
        return formula_id in self._formulas

    def __getitem__(self, formula_id: str) -> Formula:
        return self.get(formula_id)

    def __len__(self) -> int:
        return len(self._formulas)


# ── Canonical seed data ──────────────────────────────────────────────────────
# Edit formulas ONLY here. Every symbol/subscript/superscript/Greek letter/unit
# is part of the canonical object and is validated by formula_validator.py.

_SEED: List[Formula] = [
    # ---- Worked example -----------------------------------------------------
    Formula(
        id="area_triangle",
        latex=r"A = \frac{1}{2} b h",
        label="Area of a Triangle",
        description="Half the base times the height.",
        plaintext="A = (1/2) * b * h",
        symbols=(
            Symbol("area",     "A", "area",   "variable", "the triangle's area", "unit\u00b2"),
            Symbol("side_b",   "b", "base",   "variable", "length of the base",  "unit"),
            Symbol("height",   "h", "height", "variable", "perpendicular height", "unit"),
        ),
    ),
    # ---- Superscripts -------------------------------------------------------
    Formula(
        id="pythagorean_theorem",
        latex=r"c^{2} = a^{2} + b^{2}",
        label="Pythagorean Theorem",
        description="In a right triangle, the square of the hypotenuse equals "
                    "the sum of the squares of the two legs.",
        plaintext="c^2 = a^2 + b^2",
        symbols=(
            Symbol("hypotenuse", "c", "hypotenuse", "variable", "longest side", "unit"),
            Symbol("side_a",     "a", "leg a",      "variable", "first leg",    "unit"),
            Symbol("side_b",     "b", "leg b",      "variable", "second leg",   "unit"),
        ),
    ),
    # ---- Greek letter + superscript ----------------------------------------
    Formula(
        id="circle_area",
        latex=r"A = \pi r^{2}",
        label="Area of a Circle",
        description="Pi times the radius squared.",
        plaintext="A = pi * r^2",
        symbols=(
            Symbol("area",   "A",     "area",   "variable", "the circle's area",      "unit\u00b2"),
            Symbol("pi",     r"\pi",  "pi",     "constant", "ratio of circumference to diameter", ""),
            Symbol("radius", "r",     "radius", "variable", "distance from centre to edge",       "unit"),
        ),
    ),
    # ---- Subscript (label) + Greek operator + units ------------------------
    Formula(
        id="average_velocity",
        latex=r"v_{\mathrm{avg}} = \frac{\Delta x}{\Delta t}",
        label="Average Velocity",
        description="Change in position divided by change in time.",
        plaintext="v_avg = (delta x) / (delta t)",
        symbols=(
            Symbol("velocity", "v",       "average velocity", "variable", "mean rate of change of position", "m/s"),
            Symbol("delta",    r"\Delta", "delta",            "operator", "change in a quantity",            ""),
            Symbol("position", "x",       "position",         "variable", "location along an axis",          "m"),
            Symbol("time",     "t",       "time",             "variable", "elapsed time",                    "s"),
        ),
    ),
]


def load_ledger() -> FormulaLedger:
    """Return the canonical, deterministic, in-memory formula ledger."""
    return FormulaLedger(_SEED)


__all__ = ["Symbol", "Formula", "FormulaLedger", "load_ledger"]
