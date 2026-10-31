# report/: PDF of Argon's write-up with repository data

- `paper.pdf`: single-column rendering (reportlab; LaTeX was not available when it was built). The last page is an appendix, not part of the manuscript, listing data provenance and where the draft differs from the repo notes.
- `paper.tex` + `figures/*.pdf`: the same text as revtex4-2 source for Overleaf. **Not compiled yet.** `\note{...}` (red) marks the places that differ from the repo notes; delete them once resolved.
- `paper_text.py`: the manuscript text (single source for both outputs). `build.py` builds `paper.pdf`/`paper.tex`; `make_figures.py` builds `figures/`.
- `data/`: `dg_fcc_bcc.csv` (copy of docs/data from PR #2), `dft_hull_sheet.csv` (columns 1-15, rows 0-4 of `data-h3o-convex-hull.csv`), `tdi_summary_values.csv` (numbers typed in from `docs/CURRENT_ISSUES.md` and `HANDOFF.md` on the PR #2 branch).

Rebuild: `python3 report/make_figures.py && python3 report/build.py` (needs matplotlib, reportlab, pillow).

## What is real and what is a placeholder

| Figure | Status |
|---|---|
| Fig. 1 | a: real (lambda-range split of the n=1 insertion integral, 500 GPa/6000 K). b-d: placeholders |
| Fig. 2 | real but sparse: n=1,2,4 insertion only at 500 GPa/6000 K; direct H2.25O and fit extrapolation at three states |
| Fig. 3 | all placeholders |
| Fig. 4 | a, b: DFT hull sheet. c-f (MLP hull, G_liquid - G_fcc): placeholders |
| Ext. Data Fig. 2 | real (fcc-bcc benchmark vs Wilson 2013 / Cheng 2021) |

The raw TDI runs (`runs/tdi*` on Young) and `docs/data/tdi_results.csv` are not in git, which is why most panels are placeholders. Once `summarize.py runs/tdi runs/tdi_prep -o docs/data/tdi_results.csv` is committed, `make_figures.py` can be pointed at it.

## Draft vs repo notes (details in the PDF appendix, A1-A9)

- Fig. 1 text: draft puts +2,920 meV in lambda in [0.1,1]; repo has about 2,950 meV total and about 2,480 meV in that range.
- Validation metrics (1.65 GPa, 19.6 meV/atom RMSE) are not in the repo notes (they record MAE 18.4 meV/atom, force 511 meV/A, stress 2.05 GPa).
- "Lies directly on the extrapolation": repo shows +141 direct vs +55 fit (500/5500), +76 vs -83 (500/6000), +192 vs +275 (400/5000).
- The 40 meV/atom melting depression has no data in the repo.
- The first-pass dilute-limit fit points to an enthalpy-driven (dh about -0.8 eV), not entropy-driven, picture.
- Not stated in the draft: classical nuclei, no MLP-to-DFT correction, statistical errors only, open issues D1-D6 in `docs/CURRENT_ISSUES.md`.
- The draft is cut off mid-sentence in the first of four consequences; the reference list was filled in from memory.
