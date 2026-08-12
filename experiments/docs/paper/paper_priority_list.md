# Priority list of related papers for the sparse-view 3DGS regime study

## Tier 1: Must-read for the core regime story

1. ReSplat
   - Why: directly relevant to sparse-view per-scene optimization failure and feed-forward vs optimization crossover.
   - Focus: sparse-view setup, overfitting behavior, and the regime where optimization becomes harmful.

2. 3D Gaussian Splatting (Kerbl et al.)
   - Why: foundational optimization dynamics, densification, opacity reset, and refinement behavior.
   - Focus: how optimization changes geometry over time and where failure emerges.

3. MVSplat
   - Why: feed-forward sparse-view baseline with a clear inference pipeline.
   - Focus: input assumptions, view support, and evaluation protocol.

4. DepthSplat
   - Why: another strong feed-forward baseline with explicit depth-based representation.
   - Focus: how depth priors affect sparse-view reconstruction quality.

## Tier 2: Important for mechanism and failure analysis

5. SparseGS / FSGS
   - Why: sparse-view specialized optimization baselines.
   - Focus: how they stabilize optimization under limited input.

6. Diff3R
   - Why: relevant to test-time optimization and geometry degradation under sparse contexts.
   - Focus: overfitting and failure modes in optimization-based refinement.

7. PixelSplat / NoPoSplat / GS-LRM-style methods
   - Why: broader feed-forward family for comparison and positioning.
   - Focus: what they do well and where they fail under sparse conditions.

## Tier 3: Useful for broader positioning

8. InstantNGP / NeRF-based sparse-view works
   - Why: useful for contrasting 3DGS-specific behavior with earlier neural reconstruction methods.
   - Focus: sparse-view regularization and optimization behavior.

9. DUSt3R / MASt3R / VGGT-related works
   - Why: useful for initialization quality and geometry prior discussions.
   - Focus: how strong geometry priors affect optimization and refinement.

## Recommended reading order

1. 3D Gaussian Splatting
2. MVSplat
3. DepthSplat
4. ReSplat
5. SparseGS / FSGS
6. Diff3R
7. PixelSplat / NoPoSplat
8. VGGT/DUSt3R-related works
