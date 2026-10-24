# Figure contract

Question: does shared/private residual search improve the accuracy/complexity
tradeoff beyond applying the same compact selector to independent search?
The final claim follows the complete frozen results and advancement gates.

Use one quantitative grid: three dataset columns, paired NRMSE differences in
the top row and paired shared-DAG complexity differences in the bottom row.
The reference throughout is independent_compact. Four comparison methods show
the original joint control, the independent penalty selector, hybrid with the
penalty selector and hybrid with the compact selector. This separates the new
search structure from validation size selection.

Show every one of the 15 paired fold/seed cells as a point, plus its mean and
one descriptive population SD. These correlated cells do not form an IID
confidence interval. An incomplete group gets no partial mean. No data are
excluded for appearance. Negative values favor the named comparison method.

Use the task's existing Python/matplotlib environment exclusively. Export at
190 x 132 mm, editable SVG and PDF, plus 300-dpi PNG; base text is 7 pt with a
5-pt glyph floor. This is a repository figure with no journal submission target.
Require 1.5-pt strict rendered alignment, source validation, PDF glyph and
collision audits, followed by visual review. Preserve source CSV and QA records.
