# Figure and evidence QA

The two figures were rendered from the complete 108-run analysis, then inspected
panel by panel and as complete 190 x 132 mm figures. All six fold/seed cells
contribute to every nonempty plotted aggregate. The integer-cap grid is 0..149;
unattainable caps remain blank. No observations were removed to improve visual
appearance. Outer curves retain nonmonotonic changes.

Both figures passed:

- Final rendered axes alignment, including equal widths/heights and repeated
  gutters, at 1.5-point tolerance with strict mode.
- PDF text audit: minimum rendered glyph size 7 points, no glyph below 5 points.
- PDF collision audit: zero FAIL and zero WARN findings.
- Visual review: labels, legends, curves and uncertainty are readable; mean
  lines use color and line-style distinctions; shading uses the same one-SD
  rule throughout. Both CPU rows share each dataset's y scale in the absolute
  error figure. Paired differences use their own clearly marked scales.

The source validator reports 18 PASS, three WARN and zero FAIL findings. The
three warnings are intentional export choices under `FIGURE_CONTRACT.md`:

1. PNG previews accompany editable SVG and local PDF, with no TIFF submission
   file because this repository figure has no journal submission target.
2. PNG resolution is the contract's 300 dpi, above the required floor but below
   the validator's preferred 600 dpi default.
3. Width is the contract's 190 mm, rather than a journal-specific 89/183 mm.

The alignment JSON is beside each figure. Source, PDF glyph and collision
reports are in this directory. Local PDFs and collision-overlay PDFs remain
ignored; published SVG and PNG files are the viewing artifacts. PDF audit hashes
are retained in `VERIFICATION.json`.

Reaggregation of saved evidence into a second directory produced byte-identical
CSV, summary JSON and Markdown table outputs (five files). This check tests
analysis reproducibility; it does not claim CPU-limited retraining is bitwise
reproducible on another machine. The full 126-test suite passed after the new
analysis code was complete. Frozen-file checks cover all 22 diagnostic inputs
and all 117 v3 frozen inputs/models; none changed.

Publication checks additionally compare all 627 new diagnostic/figure artifact
blobs with their working-file bytes. Scoped Git attributes preserve the frozen
bytes and pin source inputs to LF. Six representative source/model/figure files
also survive an automatic-CRLF Windows checkout byte for byte. The older v3
hash check above concerns the original local frozen evidence; this change does
not rewrite legacy publication files or their manifests.
