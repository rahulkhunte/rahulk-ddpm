"""
pedagogy — exact-correctness infrastructure for educational formula animations.

Modules:
  formula_ledger     canonical, immutable store of formulas + symbols
  scene_schema       storyboard schema that references formulas by id only
  formula_validator  deterministic correctness gate (latex/symbol/scene rules)
"""

from pedagogy.formula_ledger import (
    Symbol, Formula, FormulaLedger, load_ledger,
)
from pedagogy.scene_schema import (
    STEP_KINDS, SceneStep, Scene, build_area_triangle_scene,
)
from pedagogy.formula_validator import (
    Issue, validate_latex, extract_atoms,
    validate_formula, validate_ledger, validate_scene, validate_all,
)

__all__ = [
    "Symbol", "Formula", "FormulaLedger", "load_ledger",
    "STEP_KINDS", "SceneStep", "Scene", "build_area_triangle_scene",
    "Issue", "validate_latex", "extract_atoms",
    "validate_formula", "validate_ledger", "validate_scene", "validate_all",
]
