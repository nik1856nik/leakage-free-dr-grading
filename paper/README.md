# Paper: Leakage-Free Reproduction and Validation-Selected Ensembling for DR Grading

IEEE journal format (`IEEEtran`).

**Authors:** Akshai Krishna KP, Mohammed Azhar Sait, Karthik Krishna K, Rohit A. Individual
contributions are on the last page.

| File | Content |
|---|---|
| `main.pdf` | Compiled paper (14 pages, 17 figures, 8 tables) |
| `main.tex` | Source |
| `references.bib` | 51 references |
| `figures/` | Vector PDF figures (+ PNG previews) |
| `make_figures.py`, `make_figures_extra.py` | Regenerate all figures from the project results |

## Build locally (installed: Tectonic 0.15 in `~/.local/bin`)
```
cd DR_Detection/paper
~/.local/bin/tectonic main.tex          # runs LaTeX + BibTeX as needed and writes main.pdf
```
Regenerate the figures first if any result changed:
```
uv run python paper/make_figures.py && uv run python paper/make_figures_extra.py
```
Tectonic is XeTeX-based and downloads LaTeX packages on first use, so it needs an internet
connection once; after that it uses its cache (`~/.cache/Tectonic`).

The paper also compiles with standard TeX Live (`pdflatex main && bibtex main && pdflatex main &&
pdflatex main`) and on Overleaf (zip `main.tex`, `references.bib` and `figures/` and upload).

Before submission, fill in the affiliation and corresponding-author e-mail in the `\thanks{}` line.

## Black-and-white print version: `main(bw).pdf`
Pure grayscale and print-safe. Plot series are told apart by line style and hatching, fundus photos
are brightened (CLAHE + gamma), and Grad-CAM is drawn as outlines. Rebuild:
```
uv run python paper/make_figures_bw.py            # -> figures_bw/
cd paper && ~/.local/bin/tectonic main_bw.tex
gs -q -dNOPAUSE -dBATCH -dSAFER -sDEVICE=pdfwrite -sColorConversionStrategy=Gray \
   -dProcessColorModel=/DeviceGray -dEmbedAllFonts=true -sOutputFile="main(bw).pdf" main_bw.pdf
```
`main_bw.tex` is `main.tex` with `\graphicspath{{figures_bw/}}` and one caption adapted ("bold titles"
instead of "red titles"). Re-create it after editing `main.tex`.
