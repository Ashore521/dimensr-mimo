# Figure QA

The six-panel figure shows paired differences relative to independent_compact
for all 15 fold/seed cells in each dataset and comparison. All four comparison
methods are present; the control contributes to every paired difference. No
data were omitted. Every point is visible or overlaps at its true value; small
deterministic horizontal offsets distinguish cells without changing the values.
Black diamonds and bars indicate means and descriptive population SD, not IID
confidence intervals.

The exported 190 x 132 mm figure passed strict 1.5-point rendered panel alignment,
the 5-point glyph floor (minimum found: 7 points), and collision checks with
zero FAIL and zero WARN findings. Panel-by-panel and full-figure inspection
confirmed readable labels, uncropped points, distinct method colors, and clear
zero reference lines. Vertical scales differ by dataset and metric and are
shown on every panel.

Source validation reports 18 PASS, three WARN and zero FAIL. The three warnings
match the figure contract: PNG instead of TIFF, 300-dpi previews instead of the
validator's preferred 600 dpi, and 190-mm width instead of journal-specific
89/183-mm defaults. There is no journal submission target. Editable SVG and
local PDF accompany the PNG; PDF and marked QA overlays are ignored by Git.

CSV source, plotting code, environment versions, alignment JSON and audit
reports are preserved. Local PDF hashes are retained in VERIFICATION.json.
