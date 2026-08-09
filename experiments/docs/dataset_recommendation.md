# Dataset recommendation for the sparse-view 3DGS regime study

## Recommended setup

1. Primary benchmark: RE10K
   - Best default choice for the main regime-map experiments.
   - Reason: wide availability, multi-view structure, and good fit for sparse-view evaluation.

2. External geometry validation: DTU
   - Best choice for depth/geometry-oriented analysis.
   - Reason: GT geometry is available, making it suitable for depth error, Chamfer, floater, and free-space opacity analysis.

3. Secondary candidate: DL3DV
   - Good if additional diversity is needed.
   - Reason: richer scenes and stronger geometry variation, though it is less universally available than RE10K.

## Practical recommendation

- Start with RE10K + DTU for the first full study.
- Add DL3DV only if the primary pipeline is stable and you want a stronger diversity check.
- For a lighter first pass, use 5-10 scenes from RE10K and 5-8 scenes from DTU.

## Why this is a good starting point

- RE10K supports the main regime comparison across view count, overlap, and budget.
- DTU enables failure analysis and geometry-based validation, which is important for the paper's mechanism section.
- This split keeps the main claim focused on regime behavior while still providing a stronger external validation track.
