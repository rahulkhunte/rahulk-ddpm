"""
formula_validator.py — exact-correctness gate for the formula ledger and scenes.

The validator is deterministic and local. It refuses to let a formula or scene
through unless it passes EVERY rule below. The render pipeline runs this first
and aborts on any error, so nothing incorrect is ever displayed.

Validation rules
----------------
L1  INVALID_LATEX        Formula.latex must parse/typeset under matplotlib
                         mathtext (a local LaTeX subset). Malformed LaTeX is
                         rejected.
S1  UNDECLARED_SYMBOL    Every symbol atom that appears in the LaTeX must be
                         declared in Formula.symbols (catches symbol drift).
S2  MISSING_SYMBOL       Every declared symbol must actually appear in the
                         LaTeX (no stale/missing symbols).
S3  CASING_DRIFT         A LaTeX atom that matches a declared token only when
                         case is ignored is flagged (inconsistent casing).
G1  SYMBOL_ID_DRIFT      A global symbol id must map to exactly one canonical
                         token across the whole ledger (e.g. concept "side_b"
                         is always "b", never "B").
D1  DUPLICATE_SYMBOL_ID  A formula must not declare the same symbol id twice.
SC1 SCENE_UNKNOWN_FORMULA  A scene step references a formula id absent from the
                           ledger.
SC2 SCENE_RAW_TEXT       A scene title contains LaTeX ("$" or "\\"), i.e. it is
                         trying to bypass the ledger.
SC3 SCENE_BAD_KIND       A scene step uses an unknown kind.
SC4 SCENE_UNKNOWN_SYMBOL A highlighted symbol id is not part of the referenced
                         formula.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from pedagogy.formula_ledger import Formula, FormulaLedger, load_ledger
from pedagogy.scene_schema import STEP_KINDS, Scene, build_area_triangle_scene


# Greek macro names → treated as canonical symbol atoms (token == "\name").
GREEK = {
    "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta",
    "theta", "vartheta", "iota", "kappa", "lambda", "mu", "nu", "xi", "pi",
    "varpi", "rho", "varrho", "sigma", "varsigma", "tau", "upsilon", "phi",
    "varphi", "chi", "psi", "omega",
    "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon",
    "Phi", "Psi", "Omega",
}

# Macros whose brace argument is a text LABEL, not symbols (skip their contents).
LABEL_MACROS = {"mathrm", "text", "operatorname", "mathbf", "mathit",
                "mathsf", "mathcal", "mathtt", "textrm"}


@dataclass(frozen=True)
class Issue:
    severity: str   # "error"
    code:     str
    where:    str
    message:  str

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.code} @ {self.where}: {self.message}"


# ── LaTeX validity (rule L1) ──────────────────────────────────────────────────
def validate_latex(latex: str) -> Tuple[bool, Optional[str]]:
    """Return (ok, error). Uses matplotlib mathtext — fully local & deterministic."""
    fig = Figure()
    FigureCanvasAgg(fig)
    fig.text(0.5, 0.5, f"${latex}$")
    try:
        fig.canvas.draw()
        return True, None
    except Exception as exc:  # mathtext raises ValueError on malformed input
        msg = str(exc).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__
        return False, msg


# ── Symbol extraction ─────────────────────────────────────────────────────────
def extract_atoms(latex: str) -> Set[str]:
    """
    Deterministically extract the set of canonical symbol atoms from LaTeX.

      - single ASCII letters            -> "x"
      - Greek macros                    -> "\\pi", "\\Delta", ...
      - contents of \\mathrm{...} etc.  -> ignored (these are labels)
      - structural/operator macros      -> ignored (\\frac, \\cdot, ...)
      - digits, operators, braces, _, ^ -> ignored
    """
    atoms: Set[str] = set()
    i, n = 0, len(latex)
    while i < n:
        ch = latex[i]
        if ch == "\\":
            j = i + 1
            while j < n and latex[j].isalpha():
                j += 1
            name = latex[i + 1:j]
            if name in LABEL_MACROS:
                # skip optional spaces then the balanced {...} label argument
                k = j
                while k < n and latex[k] == " ":
                    k += 1
                if k < n and latex[k] == "{":
                    depth = 0
                    while k < n:
                        if latex[k] == "{":
                            depth += 1
                        elif latex[k] == "}":
                            depth -= 1
                            if depth == 0:
                                k += 1
                                break
                        k += 1
                    i = k
                    continue
                i = j
                continue
            if name in GREEK:
                atoms.add("\\" + name)
            # any other macro is structural/operator → ignored
            i = j if j > i + 1 else i + 1
            continue
        if ch.isalpha():
            atoms.add(ch)
        i += 1
    return atoms


# ── Per-formula validation (L1, S1, S2, S3, D1) ───────────────────────────────
def validate_formula(formula: Formula) -> List[Issue]:
    issues: List[Issue] = []
    where = f"formula:{formula.id}"

    ok, err = validate_latex(formula.latex)
    if not ok:
        issues.append(Issue("error", "INVALID_LATEX", where,
                            f"LaTeX failed to typeset: {err!r} | latex={formula.latex!r}"))
        # Symbol checks are unreliable on un-parseable LaTeX; stop here.
        return issues

    # D1 — duplicate symbol ids within the formula
    seen_ids: Set[str] = set()
    for s in formula.symbols:
        if s.id in seen_ids:
            issues.append(Issue("error", "DUPLICATE_SYMBOL_ID", where,
                                f"symbol id declared twice: {s.id!r}"))
        seen_ids.add(s.id)

    declared = {s.token for s in formula.symbols}
    atoms = extract_atoms(formula.latex)

    declared_lower = {d.lower(): d for d in declared}
    atom_lower = {a.lower(): a for a in atoms}

    # S1 / S3 — atoms present in LaTeX but not declared (drift / casing)
    for a in sorted(atoms):
        if a in declared:
            continue
        if a.lower() in declared_lower:
            issues.append(Issue("error", "CASING_DRIFT", where,
                                f"LaTeX uses {a!r} but ledger declares "
                                f"{declared_lower[a.lower()]!r} (case mismatch)"))
        else:
            issues.append(Issue("error", "UNDECLARED_SYMBOL", where,
                                f"symbol {a!r} appears in LaTeX but is not declared"))

    # S2 — declared symbols missing from LaTeX
    for s in formula.symbols:
        if s.token in atoms:
            continue
        if s.token.lower() in atom_lower:
            # casing already reported under S3 from the atom side; note the gap too
            issues.append(Issue("error", "CASING_DRIFT", where,
                                f"declared token {s.token!r} (id={s.id}) not found; "
                                f"LaTeX has {atom_lower[s.token.lower()]!r}"))
        else:
            issues.append(Issue("error", "MISSING_SYMBOL", where,
                                f"declared symbol {s.token!r} (id={s.id}) "
                                f"does not appear in LaTeX"))
    return issues


# ── Ledger-wide validation (adds G1) ──────────────────────────────────────────
def validate_ledger(ledger: FormulaLedger) -> List[Issue]:
    issues: List[Issue] = []
    for f in ledger.formulas():
        issues.extend(validate_formula(f))

    # G1 — a global symbol id must map to exactly one token across the ledger
    id_to_tokens: dict = {}
    for f in ledger.formulas():
        for s in f.symbols:
            id_to_tokens.setdefault(s.id, {}).setdefault(s.token, []).append(f.id)
    for sid, tokmap in sorted(id_to_tokens.items()):
        if len(tokmap) > 1:
            detail = "; ".join(
                f"{tok!r} in [{', '.join(sorted(fs))}]" for tok, fs in sorted(tokmap.items())
            )
            issues.append(Issue("error", "SYMBOL_ID_DRIFT", f"symbol_id:{sid}",
                                f"concept maps to multiple tokens: {detail}"))
    return issues


# ── Scene validation (SC1–SC4) ────────────────────────────────────────────────
def validate_scene(scene: Scene, ledger: FormulaLedger) -> List[Issue]:
    issues: List[Issue] = []
    where = f"scene:{scene.id}"

    # SC2 — scene title must be prose, never LaTeX
    if "$" in scene.title or "\\" in scene.title:
        issues.append(Issue("error", "SCENE_RAW_TEXT", where,
                            f"title must not contain LaTeX: {scene.title!r}"))

    for idx, step in enumerate(scene.steps):
        sloc = f"{where}:step[{idx}]"
        if step.kind not in STEP_KINDS:
            issues.append(Issue("error", "SCENE_BAD_KIND", sloc,
                                f"unknown step kind {step.kind!r}"))
        if step.formula_id not in ledger:
            issues.append(Issue("error", "SCENE_UNKNOWN_FORMULA", sloc,
                                f"references unknown formula id {step.formula_id!r}"))
            continue
        formula = ledger.get(step.formula_id)
        valid_symbol_ids = {s.id for s in formula.symbols}
        for hs in step.highlight_symbols:
            if hs not in valid_symbol_ids:
                issues.append(Issue("error", "SCENE_UNKNOWN_SYMBOL", sloc,
                                    f"highlight symbol id {hs!r} not in formula "
                                    f"{step.formula_id!r}"))
    return issues


def validate_all(ledger: FormulaLedger, scenes: List[Scene]) -> List[Issue]:
    issues = validate_ledger(ledger)
    for sc in scenes:
        issues.extend(validate_scene(sc, ledger))
    return issues


def _report(issues: List[Issue]) -> None:
    if not issues:
        return
    # deterministic ordering for reproducible output
    for iss in sorted(issues, key=lambda x: (x.where, x.code, x.message)):
        print(str(iss))


def main() -> int:
    ledger = load_ledger()
    scenes = [build_area_triangle_scene()]
    issues = validate_all(ledger, scenes)

    print(f"Ledger: {len(ledger)} formula(s) -> {', '.join(ledger.ids())}")
    print(f"Scenes: {len(scenes)} -> {', '.join(s.id for s in scenes)}")

    if issues:
        print(f"\nFAILED — {len(issues)} issue(s):")
        _report(issues)
        return 1

    print("\nPASSED — all formulas and scenes are exactly correct.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())


__all__ = [
    "Issue", "validate_latex", "extract_atoms", "validate_formula",
    "validate_ledger", "validate_scene", "validate_all", "main",
]
