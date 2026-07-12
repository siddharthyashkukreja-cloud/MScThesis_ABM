# Thesis Review — Checklist of Mistakes & Changes

Full pass over every chapter (`chapters/*.tex` + `main.tex` + `references.bib`). Items are grouped
by importance. Locations are by chapter and a quoted phrase (line numbers drift as you edit).

**Good news up front:** every `\cite` resolves and every `\ref` has a `\label` — no undefined
citations or broken cross-references. All parameter values in the appendix match the code/calibration
outputs. The two genuinely-unfinished pieces are the **abstract** and the **acknowledgements** (both
empty files).

---
---

## 2. MEDIUM — consistency, citations, cross-references, content nuance
.

- [ ] **Bare `\ref`s in the Calibration chapter.** *"reported in \ref{sec:cal-synth}"*, *"\ref{tab:cal-data}"*,
  *"reported in the \ref{ch:appendix}"* render as a bare number. Add the word and a `~`: "Section~\ref{...}",
  "Table~\ref{...}", "Appendix~\ref{...}" (and drop the stray "the").

---

## 3. LOW — typos, grammar, style, spacing, cleanup

- [ ] **Missing closing parenthesis** (Results §5.1, Capital Ratios): *"(Figure~\ref{fig:res-kappa},"* — add `)`.
- [ ] **Typos / grammar:**
  - "guaranties" → "guarantees" (Model §Central Clearing Layer).
  - "Each stressed seed produce" → "produces" (Results §5.1).
  - "The trading strategies of each trader and explained below" → "are explained" (Model §Trading Agents).
  - "upto" → "up to" (Related Work "upto sixteen clients"; Results "ramping upto").
  - "it's ability" → "its ability" (Results, final paragraph).
  - stray "**S**" at the end of a table note (Calibration, Table "cal-data" notes: *"bps = 10⁻⁴. S"*).
  - "Mean-Coverage ratio" → "Moment Coverage Ratio / MCR" (Calibration, synthetic-validation paragraph).
  - "as little as three" → "as few as three" (Related Work — clients are countable).
  - "2/3rds" → "two-thirds" (Results §Synthetic).
  - "\$ 12.50" → "\$12.50" (Model §LOB — stray space after `\$`).
  - ":-" colon-dash → ":" (Model, e.g. "presented below:-", "from the FV signal:-").
- [ ] **Comma splices** to tidy: Results opening *"...same world on repeat, the major elements..."*;
  Model *"presents the design ... :-"* etc.
  


## 4. TO-DO — unfinished pieces

- [ ] **Abstract** (`chapters/00_abstract.tex`) — **empty.** (Known.)
- [ ] **Acknowledgements** (`chapters/a_acknowledgements.tex`) — **also empty** (you said "just the abstract,"
  but this one is blank too).
- [ ] **Title page** carries a *"% Replace with final thesis title"* comment — confirm the title is final.
- [ ] **References** — all resolve and styles are consistent (`plainnat`). Remaining checks are *content*:
  the modal/median wording, the ESRB-COVID swap, and the Paddrik "periphery" claim above. Also double-check
  `ofr_2026`'s "26-03 (2026)" working-paper number/year is the version you intend to cite.

---

### Quick-win order (highest value, least effort)
1. Fix the mechanism sentence (members → clients) — it's the core finding.
2. "client default" → "member default" in the intro (Aas).
3. NBCM "comfortable 8%" reword.
4. modal/median + ESRB-COVID citation + "loss θ" symbol.
5. "Section" → "Chapter" + bare `\ref`s.
6. Appendix `p_d`→`p_zi` and symbol unification.
7. Typo sweep + delete dev comments.
8. Write the abstract and acknowledgements.
