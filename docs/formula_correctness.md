# Formula Correctness Infrastructure

Exact-correctness infrastructure for educational animations. It guarantees that
**every formula a learner sees is byte-for-byte the formula that was authored and
reviewed** — never paraphrased, re-typeset, or heuristically rewritten.

The whole system is **deterministic and local**: no network, no external LaTeX
binary. LaTeX is parsed and rendered with matplotlib's built-in `mathtext`, and
GIFs are assembled with Pillow.

> This subsystem is independent of the diffusion code and does not modify it.

---

## 1. Canonical Formula Policy

1. **Formulas are canonical text objects.** A formula's LaTeX, label, symbols,
   subscripts, superscripts, Greek letters, and units are authored once in the
   ledger and are immutable (`@dataclass(frozen=True)`).
2. **Never paraphrase or rewrite.** No code may regenerate, "clean up", or
   re-typeset a formula. Renderers display `Formula.latex` verbatim.
3. **Reference by id, never by text.** Scenes and pipelines refer to a formula
   by its `id` (e.g. `area_triangle`). Raw LaTeX never appears in a scene.
4. **Validate before display.** The render pipeline runs the validator first and
   aborts on any issue, so an incorrect formula can never reach the screen.

---

## 2. Ledger Structure

File: `pedagogy/formula_ledger.py`

### `Symbol` (frozen)
| field | meaning |
|-------|---------|
| `id` | global concept id, shared across formulas (e.g. `side_b`) |
| `token` | exact LaTeX atom as written, e.g. `b`, `A`, `\pi`, `\Delta` |
| `name` | short human name (`base`) |
| `kind` | `variable` \| `constant` \| `greek` \| `operator` |
| `description` | prose gloss (no LaTeX) |
| `unit` | unit string (`m`, `s`, `m/s`, `unit²`); empty if dimensionless |

### `Formula` (frozen)
| field | meaning |
|-------|---------|
| `id` | stable key referenced by scenes (`area_triangle`) |
| `latex` | canonical LaTeX, rendered verbatim |
| `label` | human title, prose (`Area of a Triangle`) |
| `description` | prose explanation |
| `symbols` | ordered tuple of `Symbol`s appearing in `latex` |
| `plaintext` | optional canonical ASCII fallback |

### `FormulaLedger`
Immutable ordered collection keyed by `id`: `load_ledger()`, `get(id)`, `ids()`,
`formulas()`, `id in ledger`. Duplicate ids are rejected at construction.

A **global symbol id** must always map to the **same token**. For example the
concept `side_b` is `b` in both `area_triangle` and `pythagorean_theorem`. This
is what gives the casing/consistency checks teeth.

### Seeded formulas
| id | LaTeX | demonstrates |
|----|-------|--------------|
| `area_triangle` | `A = \frac{1}{2} b h` | **worked example**, fractions |
| `pythagorean_theorem` | `c^{2} = a^{2} + b^{2}` | superscripts, shared `side_b` |
| `circle_area` | `A = \pi r^{2}` | Greek letter, shared `area` |
| `average_velocity` | `v_{\mathrm{avg}} = \frac{\Delta x}{\Delta t}` | subscript label, Greek operator, units |

---

## 3. Scene Schema

File: `pedagogy/scene_schema.py`

A `Scene` is an ordered tuple of `SceneStep`s. **A step references a formula by
`formula_id` only — it carries no raw LaTeX.** The renderer pulls the canonical
text from the ledger at render time.

Step kinds (each references a ledger formula):

- `show_label` — display `Formula.label`
- `show_formula` — display `Formula.latex` (verbatim)
- `show_legend` — display the symbol legend; optional `highlight_symbols`
  (a list of **symbol ids**) restricts/emphasises rows

The only free text in a scene is `Scene.title` (prose), which the validator
checks contains no LaTeX (`$` or `\`) so it cannot smuggle in an un-ledgered
formula.

`build_area_triangle_scene()` returns the canonical worked example:
`show_label → show_formula → show_legend`, all pointing at `area_triangle`.

---

## 4. Validation Rules

File: `pedagogy/formula_validator.py`. Every rule is deterministic; output is
sorted for reproducibility. Any issue is an **error** and fails the run.

| code | rule |
|------|------|
| `INVALID_LATEX` | `Formula.latex` must typeset under matplotlib mathtext |
| `UNDECLARED_SYMBOL` | a symbol atom in the LaTeX is not declared in `symbols` (**symbol drift**) |
| `MISSING_SYMBOL` | a declared symbol does not appear in the LaTeX |
| `CASING_DRIFT` | a LaTeX atom matches a declared token only case-insensitively (**inconsistent casing**) |
| `SYMBOL_ID_DRIFT` | one global symbol id maps to more than one token across the ledger |
| `DUPLICATE_SYMBOL_ID` | a formula declares the same symbol id twice |
| `SCENE_UNKNOWN_FORMULA` | a scene step references a formula id not in the ledger |
| `SCENE_RAW_TEXT` | a scene title contains LaTeX (bypass attempt) |
| `SCENE_BAD_KIND` | a scene step uses an unknown kind |
| `SCENE_UNKNOWN_SYMBOL` | a highlighted symbol id is not part of the referenced formula |

### Symbol extraction
`extract_atoms(latex)` deterministically pulls the set of canonical atoms:

- single ASCII letters → `x`
- Greek macros → `\pi`, `\Delta`, …
- contents of `\mathrm{…}` / `\text{…}` → ignored (these are **labels**)
- structural/operator macros (`\frac`, `\cdot`, …) → ignored
- digits, operators, braces, `_`, `^` → ignored

The extracted atom set is compared (case-sensitively) against the declared
`token`s to detect drift, missing symbols, and casing problems.

### LaTeX validity
`validate_latex(latex)` typesets `$latex$` onto an Agg canvas and reports
failure if mathtext raises. This is a local LaTeX subset — no `latex`/`dvipng`
binaries required.

---

## 5. Render Workflow

File: `pipelines/render_then_animate.py`. Four deterministic stages:

1. **Validate** — `validate_all(ledger, [scene])`. On any issue, print the
   sorted report and exit non-zero; **nothing is rendered**.
2. **Render** — for each formula referenced by the scene, fetch `Formula.latex`
   from the ledger *verbatim* and typeset it to `render/<id>.png`.
3. **Animate** — replay the scene as a cumulative reveal (label → formula →
   legend), one `frames/frame_XX.png` per step, stitched into `<scene_id>.gif`.
4. **Manifest** — write `manifest.json` recording the exact LaTeX and its
   SHA-256 for every rendered formula (audit trail proving "rendered from the
   ledger exactly").

Output tree (default root `assets/pedagogy/`):

```
assets/pedagogy/area_triangle_worked_example/
├── render/area_triangle.png
├── frames/frame_00.png … frame_02.png
├── area_triangle_worked_example.gif
└── manifest.json
```

Determinism: fixed figure size/DPI, fixed fonts (`DejaVu Sans` /
`dejavusans` mathtext), white background, no timestamps; LaTeX is hashed in the
manifest so any drift is detectable byte-for-byte.

---

## 6. Worked Example — Area of a Triangle

Ledger entry (`area_triangle`):

```
A = \frac{1}{2} b h
```

| symbol | id | name | unit |
|--------|----|------|------|
| `A` | `area` | area | unit² |
| `b` | `side_b` | base | unit |
| `h` | `height` | height | unit |

The scene `area_triangle_worked_example` reveals: the label *“Area of a
Triangle”*, then the canonical formula, then the symbol legend — every beat
sourced from the ledger by id.

---

## 7. Commands

Validate the ledger and the worked-example scene:

```bash
python -m pedagogy.formula_validator
```

Render the worked example, then animate it:

```bash
python -m pipelines.render_then_animate
```

Both commands are run from the repository root.
