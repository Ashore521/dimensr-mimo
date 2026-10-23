# Figure contract

Question: do more search CPU seconds or more joint basis slots close the
prediction-error gap under the same shared-DAG complexity cap?

The conclusion will be written from the complete frozen results in
`docs/BUDGET_CAPACITY_DIAGNOSIS.md`; it is not fixed to favor any arm.

Two quantitative grids address complementary parts of this question:

1. Error versus complexity cap: three dataset columns and two CPU-budget rows
   show the absolute operating tradeoff for all three arms.
2. Paired changes: the top row shows 60 minus 20 seconds for each arm; the bottom
   row shows wide minus eight-basis joint search at each budget. Negative NRMSE
   differences favor the first condition. This separates budget and capacity
   evidence without hiding dataset heterogeneity.

Each line represents all six fixed fold/seed cells; shading is one descriptive
population SD, not a confidence interval. Missing or failed cells leave a gap.
The full integer-cap grid is shown; outer curves are neither optimized nor
smoothed. Inner-validation choices determine every plotted point. Selected
actual complexities and all per-output scores remain in source CSV files.

Use Python/matplotlib only, 190 by 132 mm, 7-point base font (no glyph below
5 points), an accessible blue/orange/green palette, editable SVG/PDF text and
300-dpi PNG previews. These are repository research figures without a journal
submission target. Publish CSV sources, scripts, dependency versions and QA
records. Require rendered panel alignment within 1.5 points, PDF glyph audit,
collision audit and visual inspection before delivery. No data exclusions are
allowed for visual convenience.
