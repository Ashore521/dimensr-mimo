# Third-party notices

The files `src/dimensr_mimo/_vendor/{units,expression,intervals}.py` are copied
unchanged from [DimenSR](https://github.com/Ashore521/dimensr), revision
`7fbcefadd105fe6266c8149cc05420de8ca74a76` (version 0.4.0), under its MIT license.
Copyright (c) 2026 DimenSR contributors. The complete license is in
`licenses/DimenSR-MIT.txt`. Their original raw-tree and interval semantics are
retained. New MIMO search, output heads, shared execution, API and CLI live outside
the vendor directory and are licensed under this repository's MIT license.

DimenSR acknowledges PhySO and Deep Symbolic Optimization as methodological
influences. This package does not include their source, datasets, article PDFs,
or the author's private proposal. It does not implement their neural policy
training. It uses NumPy, SciPy, SymPy and scikit-learn as separately licensed
dependencies installed through pip.
